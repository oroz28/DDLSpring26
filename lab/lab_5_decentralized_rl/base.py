import re
from dataclasses import dataclass, fields
from typing import cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from math_verify import parse, verify
from torch.nn.utils import clip_grad_norm_
from transformers import GenerationConfig


@dataclass
class Experience:
    sequence_ids: torch.Tensor
    advantages: torch.Tensor | None
    attention_mask: torch.Tensor | None
    action_mask: torch.Tensor
    start_ids: int
    logits_to_keep: int
    gen_log_probs: torch.Tensor

    def to(self, device: str) -> "Experience":
        members = {}

        for field in fields(self):
            v = getattr(self, field.name)

            if isinstance(v, torch.Tensor):
                v = v.to(device=device)

            members[field.name] = v

        return Experience(**members)


@dataclass
class GRPOConfig:
    num_generations: int = 12
    beta: float = 0.
    epsilon: float = 0.2
    epsilon_low: float | None = None
    epsilon_high: float | None = None
    micro_batch_size: int = 3
    steps_per_generation: int = 1
    clip_gradient: float = 1.

    def __post_init__(self):
        if self.epsilon_low == None:
            self.epsilon_low = self.epsilon

        if self.epsilon_high == None:
            self.epsilon_high = self.epsilon_low


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def extract_gsm8k(prompt_batch: dict) -> tuple[str, list, list]:
    return (
        prompt_batch["question"],
        list(map(lambda el: el.split("####")[0], prompt_batch["answer"])),
        list(map(lambda el: el.split(" ")[-1], prompt_batch["answer"]))
    )


@torch.no_grad()
def reward_answer_binary(completions, oracle_answer):
    returns = torch.zeros(len(completions), 1, dtype=torch.float)

    if not isinstance(oracle_answer, list):
        oracle_answer = [oracle_answer]

    answer_reward = torch.zeros(len(completions), 1, dtype=torch.float)
    formatting_reward = torch.zeros(len(completions), 1, dtype=torch.float)

    for i, completion in enumerate(completions):
        # search answer tag
        answer_match = re.findall(
            r"<answer>(.*?)</answer>",
            completion,
            flags=re.DOTALL,
        )
        answer = answer_match[0] if answer_match and len(
            answer_match) == 1 else None
        reward = 0.

        if answer is not None:
            formatting_reward[i] = 0.5

            if verify(parse(oracle_answer[i % len(oracle_answer)]), parse(answer)):
                answer_reward[i] = 1
                reward = 1

        if "<think>" in completion and "</think>" in completion and completion.find("</think>") > completion.find("<think>"):
            formatting_reward[i] += 0.5
        else:
            reward = 0

        if len(re.findall(r"<answer>", completion)) > 1 or len(re.findall(r"</answer>", completion)) > 1:
            reward = 0

        if len(re.findall(r"<think>", completion)) > 1 or len(re.findall(r"</think>", completion)) > 1:
            reward = 0

        extract = re.search(r'</answer>\s?', completion)

        if extract == None or extract.span()[1] != len(completion):
            reward = 0

        returns[i] = reward

    return returns, answer_reward, formatting_reward


@torch.no_grad()
def generate_rollouts(
        model, tokenizer, question: str,
        sys_prompt: str | None = None, num_rollouts=6,
        generation_config: GenerationConfig | None = None,
        is_conversational=True):
    model.eval()
    chat_messages = []

    if generation_config == None:
        generation_config = GenerationConfig(
            max_new_tokens=768,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=1.0,
            top_p=1.0,
            top_k=50,
        )

    if is_conversational:
        if sys_prompt != None:
            chat_messages.append({
                "role": "system",
                "content": sys_prompt,
            })

        chat_messages.append({
            "role": "user",
            "content": question,
        })
        model_inputs = tokenizer.apply_chat_template(
            chat_messages,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            return_dict=True,
            return_attention_mask=True
        ).to(model.device)
    else:
        model_inputs = tokenizer(
            [question],
            return_tensors="pt",
            padding=True,
            padding_side="left",
            return_attention_mask=True,
        ).to(model.device)

    # TODO: add prefix caching
    # duplicate prompt num_rollouts times
    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(
        num_rollouts, 1
    )
    model_inputs["input_ids"] = model_inputs["input_ids"].repeat(
        num_rollouts, 1)
    start_seq = model_inputs["input_ids"].shape[1]

    sequence_ids = model.generate(
        **model_inputs, generation_config=generation_config)

    completion_ids = sequence_ids[:, start_seq:]
    action_mask = (completion_ids != tokenizer.pad_token_id).long()

    return model_inputs["input_ids"], model_inputs["attention_mask"], completion_ids, action_mask


def gather(out: torch.Tensor, world_size: int) -> list[torch.Tensor]:
    # assumes outputs of shape (batch_size, sequence_length)
    # and equal batch_size across all ranks
    seq_lens = [
        torch.empty([], dtype=torch.long, device=out.device)
        for _ in range(world_size)]
    dist.all_gather(seq_lens, torch.tensor(out.size(-1), device=out.device))
    max_len = cast(int, max(seq_lens).item())
    padded_out = F.pad(out, (0, max_len - out.size(-1)))
    outs = [torch.empty_like(padded_out) for _ in range(world_size)]
    dist.all_gather(outs, padded_out.contiguous())
    outs = [out_[:, :seq_len] for seq_len, out_ in zip(seq_lens, outs)]

    return outs


@torch.no_grad()
def compute_entropy_from_logits(logits, chunk_size: int = 128) -> torch.Tensor:
    # all dims except num_classes
    original_shape = logits.shape[:-1]
    num_classes = logits.shape[-1]
    # flatten all leading dimensions into one
    flat_logits = logits.reshape(-1, num_classes)
    entropies = []

    for chunk in flat_logits.split(chunk_size, dim=0):
        logps = F.log_softmax(chunk, dim=-1)
        chunk_entropy = -(torch.exp(logps) * logps).sum(-1)
        entropies.append(chunk_entropy)

    entropies = torch.cat(entropies, dim=0)

    return entropies.reshape(original_shape)


def per_token_log_probs(logits, targets, is_logits_log=False, mem_eff=True):
    # TODO: add warning on bfloat
    if mem_eff and logits.dtype in [torch.float32, torch.float64]:
        return None
        selected_logits = torch.gather(
            logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # Shape (B, L)
        logsumexp_values = torch.stack(
            [torch.logsumexp(lg, dim=-1) for lg in logits])  # Shape (B, L)
        token_log_probs = selected_logits - logsumexp_values
    else:
        if not is_logits_log:
            logits = F.log_softmax(logits, dim=-1)

        token_log_probs = logits.gather(
            dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    return token_log_probs


@torch.no_grad()
def advantage_compute(rewards, std_scale=True):
    advantages = (rewards - rewards.mean())

    if rewards.shape[1] > 1 and std_scale:
        advantages /= (rewards.std() + 1e-8)

    return advantages


# compute the log probs
def sequences_log_probs(
        model, sequence_ids, attention_mask,
        logits_to_keep=None, batch_size=None, compute_entropy=False):
    if batch_size == None:
        batch_size = sequence_ids.shape[0]

    model_arg_logits = None

    if logits_to_keep != None:
        model_arg_logits = logits_to_keep + 1

    out = []
    entropy_out = []

    for start in range(0, sequence_ids.shape[0], batch_size):
        _loc_sequence_ids = sequence_ids[start: start + batch_size]
        _loc_attention_mask = attention_mask[start: start + batch_size]
        logits = model(
            input_ids=_loc_sequence_ids, attention_mask=_loc_attention_mask,
            use_cache=False, logits_to_keep=model_arg_logits).logits

        # remove last one (hallucinated)
        logits = logits[:, :-1, :]
        logits = logits[:, -logits_to_keep:, :]
        targets = _loc_sequence_ids[:, -logits_to_keep:]
        token_log_probs = per_token_log_probs(logits, targets)
        # take the attention mask from completion start onwards
        loss_mask = _loc_attention_mask[:, -logits_to_keep:]\
            .to(dtype=logits.dtype).contiguous()
        # TODO: not everyone does this?
        out.append(token_log_probs * loss_mask + (1.0 - loss_mask)
                   * torch.finfo(logits.dtype).min)

        if compute_entropy:
            entropy_out.append(compute_entropy_from_logits(logits, 256))

    if compute_entropy:
        return torch.cat(out, dim=0), torch.cat(entropy_out, dim=0)
    # else:
    return torch.cat(out, dim=0), None


def grpo_loss(
        log_probs, advantages, action_mask, grpo_config: GRPOConfig,
        gen_per_token_logps=None, ref_log_probs=None):
    """Compute the GRPO loss.
    """
    do_ref = True

    if gen_per_token_logps == None:
        gen_per_token_logps = log_probs.detach()
        do_ref = False

    coef_1 = torch.exp(log_probs - gen_per_token_logps)

    if do_ref:
        coef_2 = torch.clamp(
            coef_1, 1 - grpo_config.epsilon_low, 1 + grpo_config.epsilon_high)
        per_token_loss = torch.min(-coef_1 * advantages, -coef_2 * advantages)
    else:
        per_token_loss = -coef_1 * advantages

    if ref_log_probs != None:
        per_token_kl = (
            torch.exp(ref_log_probs - log_probs)
            - (ref_log_probs - log_probs)
            - 1
        )

        per_token_loss += grpo_config.beta * per_token_kl

    completion_lens = action_mask.sum(dim=-1).clamp(min=1)
    loss = (per_token_loss * action_mask).sum(dim=-1) / completion_lens

    return loss.mean()


def grpo_train_loop(
        model, optimizer, replay_buffer, grpo_config: GRPOConfig,
        compute_entropy=True):
    model.train()
    device = model.device
    mb_size = grpo_config.micro_batch_size
    grad_accum_steps = (
        len(replay_buffer) * grpo_config.num_generations // mb_size)
    update_every = grad_accum_steps // grpo_config.steps_per_generation
    loss_hist = []
    tmp_loss_hist = []
    kl_hist = []
    tmp_kl_hist = []
    entropy_hist = []
    tmp_entropy_hist = []
    _steps = 0
    optimizer.zero_grad()

    for exp in replay_buffer:
        exp: Experience
        batches = exp.sequence_ids.shape[0] // mb_size
        exp = exp.to(device)

        for mb in range(batches):
            _steps += 1
            end = (mb+1) * mb_size
            rng = (mb * mb_size, min(end, exp.sequence_ids.shape[0]))
            # compute log probs
            log_probs, entropy = sequences_log_probs(
                model,
                sequence_ids=exp.sequence_ids[rng[0]:rng[1], :],
                attention_mask=exp.attention_mask[rng[0]:rng[1], :],
                logits_to_keep=exp.logits_to_keep,
                compute_entropy=compute_entropy
            )
            # use ref log probs to compute kl-divergence
            gen_log_probs = exp.gen_log_probs[rng[0]:rng[1], :]

            with torch.no_grad():
                per_token_kl = (
                    torch.exp(gen_log_probs - log_probs)
                    - (gen_log_probs - log_probs)
                    - 1
                )

            tmp_kl_hist.append(per_token_kl.mean().item())
            tmp_entropy_hist.append(entropy.mean().item())
            del entropy
            del per_token_kl
            ref_log_probs = None
            loss = grpo_loss(
                log_probs=log_probs,
                advantages=exp.advantages[rng[0]:rng[1]],
                action_mask=exp.action_mask[rng[0]:rng[1]],
                grpo_config=grpo_config,
                ref_log_probs=ref_log_probs,
                gen_per_token_logps=gen_log_probs)

            if not loss.isfinite():
                continue

            print(f"loss={loss:.4f}")
            loss = loss / (update_every)
            tmp_loss_hist.append(loss.item())
            loss.backward()

            if _steps % update_every == 0:
                print("update")

                if grpo_config.clip_gradient != None:
                    clip_grad_norm_(
                        model.parameters(), max_norm=grpo_config.clip_gradient)
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                loss_hist.append(sum(tmp_loss_hist)/len(tmp_loss_hist))
                kl_hist.append(sum(tmp_kl_hist)/len(tmp_kl_hist))

                if compute_entropy:
                    entropy_hist.append(
                        sum(tmp_entropy_hist)/len(tmp_entropy_hist))
                tmp_entropy_hist.clear()
                tmp_loss_hist.clear()
                tmp_kl_hist.clear()

        del exp

    if _steps % update_every != 0:
        print("update")

        if grpo_config.clip_gradient != None:
            clip_grad_norm_(model.parameters(),
                            max_norm=grpo_config.clip_gradient)
            loss_hist.append(sum(tmp_loss_hist)/len(tmp_loss_hist))
            kl_hist.append(sum(tmp_kl_hist)/len(tmp_kl_hist))

            if compute_entropy:
                entropy_hist.append(
                    sum(tmp_entropy_hist) / len(tmp_entropy_hist))

        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.empty_cache()

    return loss_hist, kl_hist, entropy_hist
