Requirements:
    install pytorch suitable CUDA version
    pip install openslide-bin
    pip install monailabel
    pip install transformers
    pip install timm

Start server command: 
    monailabel start_server -a <path to pathology> -s <path to studies> -c models <model name>
    E.g: monailabel start_server --app apps/pathology --studies datasets/ -c models segmentation_region