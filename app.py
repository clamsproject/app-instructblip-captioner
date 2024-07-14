import argparse
import logging
from typing import Union
import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, BitsAndBytesConfig

from clams import ClamsApp, Restifier
from mmif import Mmif, View, Document, AnnotationTypes, DocumentTypes
from mmif.utils import video_document_helper as vdh

class InstructblipCaptioner(ClamsApp):

    def __init__(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float8,
        )
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-vicuna-7b", 
            quantization_config=quantization_config, 
            device_map="auto"
        )
        self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
        super().__init__()

    def _appmetadata(self):
        # see https://sdk.clams.ai/autodoc/clams.app.html#clams.app.ClamsApp._load_appmetadata
        # Also check out ``metadata.py`` in this directory. 
        # When using the ``metadata.py`` leave this do-nothing "pass" method here. 
        pass

    def get_prompt(self, label: str, prompt_map: dict, default_prompt: str) -> str:
        prompt = prompt_map.get(label, default_prompt)
        if prompt == "-":
            return None
        prompt = f"[INST] <image>\n{prompt}\n[/INST]"
        return prompt

    def _annotate(self, mmif: Union[str, dict, Mmif], **parameters) -> Mmif:
        label_map = parameters.get('promptMap')
        default_prompt = parameters.get('defaultPrompt')
        frame_interval = parameters.get('frameInterval', 10)
        batch_size = parameters.get('batchSize', 4)  # Default batch size

        video_doc: Document = mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0]
        input_view: View = mmif.get_views_for_document(video_doc.properties.id)[0]
        new_view: View = mmif.new_view()
        self.sign_view(new_view, parameters)
        
        timeframes = input_view.get_annotations(AnnotationTypes.TimeFrame)
        
        prompts = []
        images = []
        annotations = []

        def process_batch(prompts, images, annotations):
            inputs = self.processor(images=images, text=prompts, padding=True, return_tensors="pt")
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_length=256,
                min_length=1,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
            )
            generated_texts = self.processor.batch_decode(outputs, skip_special_tokens=True)
            for generated_text, annotation_id in zip(generated_texts, annotations):
                text_document = new_view.new_textdocument(generated_text.strip())
                alignment = new_view.new_annotation(AnnotationTypes.Alignment)
                alignment.add_property("source", annotation_id)
                alignment.add_property("target", text_document.id)

        if timeframes:
            for timeframe in timeframes:
                label = timeframe.get_property('label')
                prompt = self.get_prompt(label, label_map, default_prompt)
                if not prompt:
                    continue

                representatives = timeframe.get("representatives") if "representatives" in timeframe.properties else None
                if representatives:
                    image = vdh.extract_representative_frame(mmif, timeframe)
                else:
                    image = vdh.extract_mid_frame(mmif, timeframe)

                prompts.append(prompt)
                images.append(image)
                annotations.append({'source': timeframe.id})

                if len(prompts) == batch_size:
                    process_batch(prompts, images, annotations)
                    prompts, images, annotations = [], [], []

            if prompts:
                process_batch(prompts, images, annotations)
        else:
            total_frames = vdh.get_frame_count(video_doc)
            for frame_number in range(0, total_frames, frame_interval):
                image = vdh.extract_frames_as_images(video_doc, [frame_number], as_PIL=True)[0]
                prompt = default_prompt

                prompts.append(prompt)
                images.append(image)
                # create new timepoint annotation
                time_point = new_view.new_annotation(AnnotationTypes.TimePoint)
                time_point.add_property("timePoint", frame_number)
                annotations.append(time_point.id)

                if len(prompts) == batch_size:
                    process_batch(prompts, images, annotations)
                    prompts, images, annotations = [], [], []

            if prompts:
                process_batch(prompts, images, annotations)

        return mmif

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    parser.add_argument("--frameInterval", type=int, default=10, help="Interval of frames for captioning when no timeframes are present")
    parser.add_argument("--batchSize", type=int, default=4, help="Batch size for processing prompt+image pairs")

    parsed_args = parser.parse_args()

    app = InstructblipCaptioner()

    http_app = Restifier(app, port=int(parsed_args.port))
    if parsed_args.production:
        http_app.serve_production()
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
