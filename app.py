"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import argparse

import gradio as gr
import torch
from PIL import Image

from donut import DonutModel


def demo_process_vqa(input_img, question):
    global pretrained_model, task_prompt, task_name
    input_img = Image.fromarray(input_img)
    user_prompt = task_prompt.replace("{user_input}", question)
    output = pretrained_model.inference(input_img, prompt=user_prompt)["predictions"][0]
    return output


def demo_process(input_img):
    global pretrained_model, task_prompt, task_name
    input_img = Image.fromarray(input_img)
    output = pretrained_model.inference(image=input_img, prompt=task_prompt)["predictions"][0]
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # `task`: 모델이 특정 작업을 수행할 때 사용하는 데이터셋의 이름
    parser.add_argument("--task", type=str, default="docvqa")
    # `pretrained_path`: Hugging Face Model Hub에서 제공되는 모델 경로임. 보통 `organization/repository` 형식임. 
    parser.add_argument("--pretrained_path", type=str, default="naver-clova-ix/donut-base-finetuned-docvqa")
    # `port`, `url`, `sample_img_path`은 굳이 입력하지 않아도 기본값으로 실행됨
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--sample_img_path", type=str)
    args, left_argv = parser.parse_known_args()

    task_name = args.task
    if "docvqa" == task_name:
        task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
    else:  # rvlcdip, cord, ...
        task_prompt = f"<s_{task_name}>"

    example_sample = []
    if args.sample_img_path:
        example_sample.append(args.sample_img_path)

    # CORD(document parsing)으로 실행 중 오류 발생하여 수정함 but url에서 이미지 업로드하면 error 발생함. 이유는 해결 못함.
    pretrained_model = DonutModel.from_pretrained(args.pretrained_path, ignore_mismatched_sizes=True)
    # default) pretrained_model = DonutModel.from_pretrained(args.pretrained_path)

    if torch.cuda.is_available():
        pretrained_model.half()
        device = torch.device("cuda")
        pretrained_model.to(device)

    pretrained_model.eval()

    demo = gr.Interface(
        fn=demo_process_vqa if task_name == "docvqa" else demo_process,
        inputs=["image", "text"] if task_name == "docvqa" else "image",
        outputs="json",
        title=f"Donut 🍩 demonstration for `{task_name}` task",
        examples=[example_sample] if example_sample else None,
    )
    demo.launch(server_name=args.url, server_port=args.port)
