python video_demo.py --cfg ./cfgs/res101_ls.yml \
                     --net res101 \
                     --cuda \
                     --trained_model ./models/res101/coco/faster_rcnn_1_10_14657.pth \
                     --video_path /data1/kinetics-600/crying/-C7WEhlgf7w_000097_000107.mp4 \
                     --outdir ./outs/ \
                     --dataset coco \
                     --txt_file /usr/home/sut/Eddiework/coco-labels/coco-labels-2014_2017.txt
