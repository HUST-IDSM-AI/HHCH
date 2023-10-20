# HHCH

Implementation of Exploring hierarchical information in hyperbolic space for self-supervised image hashing

# Dataset Preparation

<table>
<tr>
<td >dataset</td><td>class_num</td><td>label type</td><td>source</td>
</tr>
<tr>
<td>ImageNet</td><td>100</td><td>single</td><td><a href="https://drive.google.com/drive/folders/0B7IzDz-4yH_HOXdoaDU4dk40RFE?resourcekey=0-yXVCpvfmjTx-OBW6PsSMiA">source</a>#</td>
</tr>
<tr>
<td>COCO</td><td>80</td><td>multi</td><td><a href="https://drive.google.com/drive/folders/0B7IzDz-4yH_HOXdoaDU4dk40RFE?resourcekey=0-yXVCpvfmjTx-OBW6PsSMiA">source</a>#</td>
</tr>
<tr>
<td>NUS-WIDE</td><td>21</td><td>multi</td><td><a href="https://drive.google.com/drive/folders/0B7IzDz-4yH_HOXdoaDU4dk40RFE?resourcekey=0-yXVCpvfmjTx-OBW6PsSMiA">source</a>#</td>
</tr>
<tr>
<td>NIRFlickr-25K</td><td>24</td><td>multi</td><td><a href="https://press.liacs.nl/mirflickr/mirdownload.html">source</a></td>
</tr>
<tr>
<td>VOC2012</td><td>20</td><td>multi</td><td><a href="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html">source</a></td>
</tr>
<tr>
<td>CIFAR-10</td><td>10</td><td>single</td><td><a href="http://www.cs.toronto.edu/~kriz/cifar.html">source</a></td>
</tr>


</table>

# Run

```shell
python main.py --hyper_c 0.1 --data_name imagenet --data_path xxxx --lambda_q 0.01 --lr 0.0001 --hash_bit 64 --batch_size 64 --R 1000 --start_eval 40 --eval_epochs 2 --epochs 60   --cluster_num 1500,1000,800 --HIC  --HPC
```

