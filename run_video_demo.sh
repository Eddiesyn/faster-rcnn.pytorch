python video_demo.py --cfg ./cfgs/res101.yml \
                     --net res101 \
                     --cuda \
                     --trained_model ./models/res101/pascal_voc/faster_rcnn_1_7_10021.pth \
                     --video_path /data1/kinetics-600/arguing/1h8TNeJeQ5s_000000_000010.mp4 \
                     --outdir ./outs/
