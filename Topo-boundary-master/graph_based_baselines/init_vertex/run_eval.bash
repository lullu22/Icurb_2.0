
python eval.py --dataset_dir ./dataset_PMM-NY
python utils/init_vertex_extraction.py
mkdir -p ./space_net_dataset/init_vertices
cp -r ./records/endpoint/vertices/* ./space_net_dataset/init_vertices/