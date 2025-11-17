cd ../src || exit

gpu=0

tta_algo='Mint'
tta_mode='single'

datasets=('CIFAR10C' 'CIFAR100C' 'ImageNetC5K')
models=('ViT-B/32' 'ViT-B/16' 'ViT-L/14')

for i in {0..2}; do
  {
    CUDA_VISIBLE_DEVICES=${gpu} python main.py \
      --dataset ${datasets[i]} \
      --model ${models[i]} \
      --tta_algo ${tta_algo} \
      --tta_mode ${tta_mode} \
      --cuda \
      --batch_size 20 \
      --save_to "../log/${datasets[i]}_${tta_mode}/${tta_algo}.csv"
  }
done
