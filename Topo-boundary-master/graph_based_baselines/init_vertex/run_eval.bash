
python eval.py --dataset_dir ./dataset_manhattan
python utils/init_vertex_extraction.py
mkdir -p ./space_net_dataset/init_vertices
cp -r ./records/endpoint/vertices/* ./space_net_dataset/init_vertices/