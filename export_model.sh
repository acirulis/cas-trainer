python export_inference_graph.py --input type image_tensor --pipeline_config_path=andis2/faster_rcnn_resnet101_coco.config --trained_checkpoint_prefix=andis2/model.ckpt-1588 --output_directory=/home/whitedigital/Andis/cas-trainer/trained_model/

cp andis2/cards-train.pbtxt /home/whitedigital/Andis/cas-trainer/trained_model

