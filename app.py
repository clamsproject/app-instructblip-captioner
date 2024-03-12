import argparse
import logging
from typing import Union
import torch
from PIL import Image
import requests
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import numpy as np
import cv2

from clams import ClamsApp, Restifier
from mmif import Mmif, View, Annotation, Document, AnnotationTypes, DocumentTypes

# For an NLP tool we need to import the LAPPS vocabulary items
from lapps.discriminators import Uri
from mmif.utils import video_document_helper as vdh

class InstructblipCaptioner(ClamsApp):

    def __init__(self):
        self.model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
        self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
        self.device = "cpu" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        super().__init__()

    def _appmetadata(self):
        # see https://sdk.clams.ai/autodoc/clams.app.html#clams.app.ClamsApp._load_appmetadata
        # Also check out ``metadata.py`` in this directory. 
        # When using the ``metadata.py`` leave this do-nothing "pass" method here. 
        pass

    def _annotate(self, mmif: Union[str, dict, Mmif], **parameters) -> Mmif:
        print ("called annotate")
        video_doc: Document = mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0]
        input_view: View = mmif.get_views_for_document(video_doc.properties.id)[0]

        new_view: View = mmif.new_view()
        self.sign_view(new_view, parameters)
        print("Starting annotation process for timeframes.")
        for timeframe in input_view.get_annotations(AnnotationTypes.TimeFrame):
            try:
                print(f"Processing timeframe: {timeframe.id}")
                if "representatives" in timeframe.properties and timeframe.properties["representatives"]:
                    representative_id = timeframe.get("representatives")[0]
                    print(f"Found representative: {representative_id}")
                    representative: AnnotationTypes.TimePoint = input_view.get_annotation_by_id(representative_id)
                    frame_index = vdh.convert(representative.get("timePoint"), "milliseconds",
                                            "frame", vdh.get_framerate(video_doc))
                    print(f"Frame index for representative: {frame_index}")
                else:
                    start_frame = timeframe.get_property("start")
                    end_frame= timeframe.get_property("end")
                    if end_frame - start_frame < 30:
                        continue
                    print(f"Calculating frame index from start {start_frame}ms and end {end_frame}ms")
                    frame_index = (start_frame + end_frame) // 2
                    print(f"Frame index calculated: {frame_index}")
                
                image: Image.Image = vdh.extract_frames_as_images(video_doc, [frame_index], as_PIL=True)[0]

                prompt = "Describe this frame from a television program."
                print(f"Using prompt: '{prompt}'")
                inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
                print("Inputs prepared for the model.")
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
                generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                print (generated_text)
                text_document = new_view.new_textdocument(generated_text)
                self.create_alignment(new_view, timeframe.id, text_document.id)

            except Exception as e:
                self.logger.error(f"Error processing timeframe: {e}")
                continue
        return mmif

    def create_alignment(self, view, source_id, target_id):
        alignment = view.new_annotation(AnnotationTypes.Alignment)
        alignment.properties.source = source_id
        alignment.properties.target = target_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    # add more arguments as needed
    # parser.add_argument(more_arg...)

    parsed_args = parser.parse_args()

    # create the app instance
    app = InstructblipCaptioner()

    http_app = Restifier(app, port=int(parsed_args.port))
    # for running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
