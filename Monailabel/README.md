Requirements:
1. install pytorch suitable CUDA version
2. pip install openslide-bin
3. pip install monailabel
4. pip install transformers
5. pip install timm

Start server command: 
monailabel start_server -a <path to pathology> -s <path to studies> -c models <model name>
E.g: monailabel start_server --app apps/pathology --studies datasets/ -c models segmentation_region
