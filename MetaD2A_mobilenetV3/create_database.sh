#bash create_database.sh all predictor 0 49

IMGNET_PATH='/w14/dataset/ILSVRC2012' # PUT YOUR ILSVRC2012 DIR

for ((ind=$2;ind<=$3;ind++))
do
  python build_database.py --gpu $1 \
               --model_name $4 \
               --index $ind \
               --imgnet $IMGNET_PATH \
               --hs 512 \
               --nz 56
done


