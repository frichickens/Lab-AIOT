Requirements:
1. install pytorch suitable CUDA version
2. pip install openslide-bin
3. pip install monailabel
4. pip install transformers
5. pip install timm

Start server command: 
monailabel start_server -a <path_to_pathology> -s <path_to_studies> -c models <model_name>
E.g: monailabel start_server -a apps/pathology -s datasets/ -c models segmentation_region
