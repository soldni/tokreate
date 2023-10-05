from typing import Any, List, Optional, Union

import together

from .base import BaseProvider, ProviderRegistry, ProviderResult


class TogetherAI(BaseProvider):
    def __init__(self, api_key: Optional[str] = None) -> None:
        together.api_key = api_key or together.api_key

        model_config: Union[dict, None] = None
        for model in together.Models.list():
            if model["name"] == self.model:
                model_config = model["config"]

        if model_config is None:
            raise ValueError(f"Model {self.model} not found.")

        self.stop_tokens = model_config["stop"]
        self.prompt_format = model_config.get("prompt_format", None)

    def complete(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        system_message: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        top_k: int = 60,
        top_p: float = 0.6,
        repetition_penalty: float = 1.1,
        **kwargs,
    ) -> ProviderResult:
        """
        Args:
            history: messages in OpenAI format, each dict must include role and content key.
            system_message: system messages in OpenAI format, must have role and content key.
              It can has name key to include few-shots examples.
        """

        if history:
            raise ValueError(f"History not supported for {self.model}")

        if system_message:
            raise ValueError(f"System message not supported for {self.model}")

        with self.track_latency() as latency:
            response: Any = together.Complete.create(
                prompt=self.prompt_format.format(prompt=prompt) if self.prompt_format else prompt,
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stop=self.stop_tokens,
                **kwargs,
            )

        output = response.pop("output")
        completion = output.pop("choices")[0]["text"]
        model_inputs = response.pop("args")
        response.pop("prompt")  # already in model_inputs
        meta = {**response, **output, "latency": latency.value}
        return ProviderResult(text=completion, inputs=model_inputs, provider=self, meta=meta)

    async def acomplete(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        system_message: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 300,
        top_k: int = 50_000,
        top_p: float = 0.6,
        repetition_penalty: float = 1.1,
        **kwargs,
    ) -> ProviderResult:
        return self.complete(
            prompt=prompt,
            history=history,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )


@ProviderRegistry.add
class AustismChronosHermes13b(TogetherAI):
    model = "Austism/chronos-hermes-13b"


@ProviderRegistry.add
class EleutherAIPythia12bV0(TogetherAI):
    model = "EleutherAI/pythia-12b-v0"


@ProviderRegistry.add
class EleutherAIPythia1bV0(TogetherAI):
    model = "EleutherAI/pythia-1b-v0"


@ProviderRegistry.add
class EleutherAIPythia28bV0(TogetherAI):
    model = "EleutherAI/pythia-2.8b-v0"


@ProviderRegistry.add
class EleutherAIPythia69b(TogetherAI):
    model = "EleutherAI/pythia-6.9b"


@ProviderRegistry.add
class GrypheMythoMaxL213b(TogetherAI):
    model = "Gryphe/MythoMax-L2-13b"


@ProviderRegistry.add
class HuggingFaceH4StarchatAlpha(TogetherAI):
    model = "HuggingFaceH4/starchat-alpha"


@ProviderRegistry.add
class NousResearchNousHermes13b(TogetherAI):
    model = "NousResearch/Nous-Hermes-13b"


@ProviderRegistry.add
class NousResearchNousHermesLlama213b(TogetherAI):
    model = "NousResearch/Nous-Hermes-Llama2-13b"


@ProviderRegistry.add
class NumbersStationNsqlLlama27B(TogetherAI):
    model = "NumbersStation/nsql-llama-2-7B"


@ProviderRegistry.add
class OpenAssistantLlama270bOasstSftV10(TogetherAI):
    model = "OpenAssistant/llama2-70b-oasst-sft-v10"


@ProviderRegistry.add
class OpenAssistantOasstSft4Pythia12bEpoch35(TogetherAI):
    model = "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"


@ProviderRegistry.add
class OpenAssistantStablelm7bSftV7Epoch3(TogetherAI):
    model = "OpenAssistant/stablelm-7b-sft-v7-epoch-3"


@ProviderRegistry.add
class PhindPhindCodeLlama34BPythonV1(TogetherAI):
    model = "Phind/Phind-CodeLlama-34B-Python-v1"


@ProviderRegistry.add
class PhindPhindCodeLlama34BV2(TogetherAI):
    model = "Phind/Phind-CodeLlama-34B-v2"


@ProviderRegistry.add
class WizardLMWizardCoder15BV10(TogetherAI):
    model = "WizardLM/WizardCoder-15B-V1.0"


@ProviderRegistry.add
class WizardLMWizardCoderPython34BV10(TogetherAI):
    model = "WizardLM/WizardCoder-Python-34B-V1.0"


@ProviderRegistry.add
class WizardLMWizardLM70BV10(TogetherAI):
    model = "WizardLM/WizardLM-70B-V1.0"


@ProviderRegistry.add
class BigcodeStarcoder(TogetherAI):
    model = "bigcode/starcoder"


@ProviderRegistry.add
class DatabricksDollyV212b(TogetherAI):
    model = "databricks/dolly-v2-12b"


@ProviderRegistry.add
class DatabricksDollyV23b(TogetherAI):
    model = "databricks/dolly-v2-3b"


@ProviderRegistry.add
class DatabricksDollyV27b(TogetherAI):
    model = "databricks/dolly-v2-7b"


@ProviderRegistry.add
class DefogSqlcoder(TogetherAI):
    model = "defog/sqlcoder"


@ProviderRegistry.add
class GarageBAIndPlatypus270BInstruct(TogetherAI):
    model = "garage-bAInd/Platypus2-70B-instruct"


@ProviderRegistry.add
class HuggyllamaLlama13b(TogetherAI):
    model = "huggyllama/llama-13b"


@ProviderRegistry.add
class HuggyllamaLlama30b(TogetherAI):
    model = "huggyllama/llama-30b"


@ProviderRegistry.add
class HuggyllamaLlama65b(TogetherAI):
    model = "huggyllama/llama-65b"


@ProviderRegistry.add
class HuggyllamaLlama7b(TogetherAI):
    model = "huggyllama/llama-7b"


@ProviderRegistry.add
class LmsysFastchatT53bV10(TogetherAI):
    model = "lmsys/fastchat-t5-3b-v1.0"


@ProviderRegistry.add
class LmsysVicuna13bV13(TogetherAI):
    model = "lmsys/vicuna-13b-v1.3"


@ProviderRegistry.add
class LmsysVicuna13bV1516k(TogetherAI):
    model = "lmsys/vicuna-13b-v1.5-16k"


@ProviderRegistry.add
class LmsysVicuna13bV15(TogetherAI):
    model = "lmsys/vicuna-13b-v1.5"


@ProviderRegistry.add
class LmsysVicuna7bV13(TogetherAI):
    model = "lmsys/vicuna-7b-v1.3"


@ProviderRegistry.add
class MistralaiMistral7BInstructV01(TogetherAI):
    model = "mistralai/Mistral-7B-Instruct-v0.1"


@ProviderRegistry.add
class MistralaiMistral7BV01(TogetherAI):
    model = "mistralai/Mistral-7B-v0.1"


@ProviderRegistry.add
class TogethercomputerCodeLlama13bInstruct(TogetherAI):
    model = "togethercomputer/CodeLlama-13b-Instruct"


@ProviderRegistry.add
class TogethercomputerCodeLlama13bPython(TogetherAI):
    model = "togethercomputer/CodeLlama-13b-Python"


@ProviderRegistry.add
class TogethercomputerCodeLlama13b(TogetherAI):
    model = "togethercomputer/CodeLlama-13b"


@ProviderRegistry.add
class TogethercomputerCodeLlama34bInstruct(TogetherAI):
    model = "togethercomputer/CodeLlama-34b-Instruct"


@ProviderRegistry.add
class TogethercomputerCodeLlama34bPython(TogetherAI):
    model = "togethercomputer/CodeLlama-34b-Python"


@ProviderRegistry.add
class TogethercomputerCodeLlama34b(TogetherAI):
    model = "togethercomputer/CodeLlama-34b"


@ProviderRegistry.add
class TogethercomputerCodeLlama7bInstruct(TogetherAI):
    model = "togethercomputer/CodeLlama-7b-Instruct"


@ProviderRegistry.add
class TogethercomputerCodeLlama7bPython(TogetherAI):
    model = "togethercomputer/CodeLlama-7b-Python"


@ProviderRegistry.add
class TogethercomputerCodeLlama7b(TogetherAI):
    model = "togethercomputer/CodeLlama-7b"


@ProviderRegistry.add
class TogethercomputerGPTJT6BV1(TogetherAI):
    model = "togethercomputer/GPT-JT-6B-v1"


@ProviderRegistry.add
class TogethercomputerGPTJTModeration6B(TogetherAI):
    model = "togethercomputer/GPT-JT-Moderation-6B"


@ProviderRegistry.add
class TogethercomputerGPTNeoXTChatBase20B(TogetherAI):
    model = "togethercomputer/GPT-NeoXT-Chat-Base-20B"


@ProviderRegistry.add
class TogethercomputerKoala13B(TogetherAI):
    model = "togethercomputer/Koala-13B"


@ProviderRegistry.add
class TogethercomputerLLaMA27B32K(TogetherAI):
    model = "togethercomputer/LLaMA-2-7B-32K"


@ProviderRegistry.add
class TogethercomputerLlama27B32KInstruct(TogetherAI):
    model = "togethercomputer/Llama-2-7B-32K-Instruct"


@ProviderRegistry.add
class TogethercomputerPythiaChatBase7BV016(TogetherAI):
    model = "togethercomputer/Pythia-Chat-Base-7B-v0.16"


@ProviderRegistry.add
class TogethercomputerQwen7BChat(TogetherAI):
    model = "togethercomputer/Qwen-7B-Chat"


@ProviderRegistry.add
class TogethercomputerQwen7B(TogetherAI):
    model = "togethercomputer/Qwen-7B"


@ProviderRegistry.add
class TogethercomputerRedPajamaINCITE7BBase(TogetherAI):
    model = "togethercomputer/RedPajama-INCITE-7B-Base"


@ProviderRegistry.add
class TogethercomputerRedPajamaINCITE7BChat(TogetherAI):
    model = "togethercomputer/RedPajama-INCITE-7B-Chat"


@ProviderRegistry.add
class TogethercomputerRedPajamaINCITE7BInstruct(TogetherAI):
    model = "togethercomputer/RedPajama-INCITE-7B-Instruct"


@ProviderRegistry.add
class TogethercomputerRedPajamaINCITEBase3BV1(TogetherAI):
    model = "togethercomputer/RedPajama-INCITE-Base-3B-v1"


@ProviderRegistry.add
class TogethercomputerRedPajamaINCITEChat3BV1(TogetherAI):
    model = "togethercomputer/RedPajama-INCITE-Chat-3B-v1"


@ProviderRegistry.add
class TogethercomputerRedPajamaINCITEInstruct3BV1(TogetherAI):
    model = "togethercomputer/RedPajama-INCITE-Instruct-3B-v1"


@ProviderRegistry.add
class TogethercomputerAlpaca7b(TogetherAI):
    model = "togethercomputer/alpaca-7b"


@ProviderRegistry.add
class TogethercomputerCodegen216B(TogetherAI):
    model = "togethercomputer/codegen2-16B"


@ProviderRegistry.add
class TogethercomputerCodegen27B(TogetherAI):
    model = "togethercomputer/codegen2-7B"


@ProviderRegistry.add
class TogethercomputerFalcon40bInstruct(TogetherAI):
    model = "togethercomputer/falcon-40b-instruct"


@ProviderRegistry.add
class TogethercomputerFalcon40b(TogetherAI):
    model = "togethercomputer/falcon-40b"


@ProviderRegistry.add
class TogethercomputerFalcon7bInstruct(TogetherAI):
    model = "togethercomputer/falcon-7b-instruct"


@ProviderRegistry.add
class TogethercomputerFalcon7b(TogetherAI):
    model = "togethercomputer/falcon-7b"


@ProviderRegistry.add
class TogethercomputerGuanaco13b(TogetherAI):
    model = "togethercomputer/guanaco-13b"


@ProviderRegistry.add
class TogethercomputerGuanaco33b(TogetherAI):
    model = "togethercomputer/guanaco-33b"


@ProviderRegistry.add
class TogethercomputerGuanaco65b(TogetherAI):
    model = "togethercomputer/guanaco-65b"


@ProviderRegistry.add
class TogethercomputerGuanaco7b(TogetherAI):
    model = "togethercomputer/guanaco-7b"


@ProviderRegistry.add
class TogethercomputerLlama213bChat(TogetherAI):
    model = "togethercomputer/llama-2-13b-chat"


@ProviderRegistry.add
class TogethercomputerLlama213b(TogetherAI):
    model = "togethercomputer/llama-2-13b"


@ProviderRegistry.add
class TogethercomputerLlama270bChat(TogetherAI):
    model = "togethercomputer/llama-2-70b-chat"


@ProviderRegistry.add
class TogethercomputerLlama270b(TogetherAI):
    model = "togethercomputer/llama-2-70b"


@ProviderRegistry.add
class TogethercomputerLlama27bChat(TogetherAI):
    model = "togethercomputer/llama-2-7b-chat"


@ProviderRegistry.add
class TogethercomputerLlama27b(TogetherAI):
    model = "togethercomputer/llama-2-7b"


@ProviderRegistry.add
class TogethercomputerMpt30bChat(TogetherAI):
    model = "togethercomputer/mpt-30b-chat"


@ProviderRegistry.add
class TogethercomputerMpt30bInstruct(TogetherAI):
    model = "togethercomputer/mpt-30b-instruct"


@ProviderRegistry.add
class TogethercomputerMpt30b(TogetherAI):
    model = "togethercomputer/mpt-30b"


@ProviderRegistry.add
class TogethercomputerMpt7bChat(TogetherAI):
    model = "togethercomputer/mpt-7b-chat"


@ProviderRegistry.add
class TogethercomputerMpt7b(TogetherAI):
    model = "togethercomputer/mpt-7b"


@ProviderRegistry.add
class TogethercomputerReplitCodeV13b(TogetherAI):
    model = "togethercomputer/replit-code-v1-3b"


@ProviderRegistry.add
class UpstageSOLAR070b16bit(TogetherAI):
    model = "upstage/SOLAR-0-70b-16bit"
