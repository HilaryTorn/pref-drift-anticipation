import pytest
from tests.helpers import create_default_config
from lcb_runner.utils import PLang
import os
import json


@pytest.mark.parametrize(
    "model_path, model_type, expected_output_dir_name",
    [
        (
            "/nfs2/models/exps/qwen3_think_fuzed/epoch_2",
            "VLLMAsync",
            "qwen3_think_fuzed_epoch_2_cot",
        ),
        (
            "/nfs2/models/qwen3_think_fuzed/ep_0_step_899",
            "VLLMAsync",
            "qwen3_think_fuzed_ep_0_step_899_cot",
        ),
        (
            "/nfs2/models/qwen3_think_fuzed/step_100",
            "VLLMAsync",
            "qwen3_think_fuzed_step_100_cot",
        ),
        ("Qwen/Qwen3-32B", "VLLMAsync", "Qwen3-32B_cot"),
    ],
)
def test_output_folder_name(
    model_path: str, model_type: str, expected_output_dir_name: str
):
    args = create_default_config(model_type=model_type, local_model_path=model_path)

    assert args.output_dir == expected_output_dir_name


@pytest.mark.parametrize(
    "data, use_cot, expected_output",
    [
        ("{}", True, {"enable_thinking": True}),
        ('{"key": 1245}', True, {"enable_thinking": True, "key": 1245}),
        ('{"enable_thinking": true}', True, {"enable_thinking": True}),
        (
            '{"preserve_previous_think":true}',
            True,
            {"enable_thinking": True, "preserve_previous_think": True},
        ),
        ('{"preserve_previous_think":true}', False, {"preserve_previous_think": True}),
        ("{}", False, {}),
    ],
)
def test_chat_template_kwargs(data: str, use_cot: bool, expected_output: dict):

    args = create_default_config(chat_template_kwargs=data, cot_code_execution=use_cot)

    assert args.chat_template_kwargs == expected_output


@pytest.mark.parametrize(
    "reasoning_kw",
    [
        ("enable_thinking"),
        ("thinking"),
    ],
)
def test_reasoning_kw(reasoning_kw: str):
    args = create_default_config(enable_reasoning_kw=reasoning_kw)
    assert isinstance(args.chat_template_kwargs, dict)
    assert reasoning_kw in args.chat_template_kwargs
    assert args.chat_template_kwargs[reasoning_kw] == True


def test_plangs_methods():

    plangs = PLang.get_all_plangs()
    for plang in plangs:
        try:
            PLang.get_display_lang_name(plang)
            PLang.get_single_line_comment(plang)
            PLang.get_markdown_lang_name(plang)
        except KeyError:
            pytest.fail(f"Error processing plang={plang}")

        match = PLang.fuzzy_match(plang)
        assert match in plangs


def test_plangs_matching():

    plang = PLang.Cpp
    assert plang == PLang.Cpp
    assert plang == "cpp"
    assert plang == "CPP"
    assert plang == "C++"
    assert plang == "c++"
    assert plang != PLang.Java
    assert plang != "java"

    # lang contains in list
    assert "cpp" in [PLang.Cpp, PLang.Csharp]
    assert "java" not in [PLang.Cpp, PLang.Csharp]

    with pytest.raises(TypeError):
        # not implemented
        assert PLang.Cpp in set([PLang.Cpp, PLang.Csharp])
