#!/bin/bash
set -e 

if [ $# -lt 2 ]; then
    echo "❌ 错误: 参数不足。"
    echo "用法示例: sh run.sh <数据集> <方法1> [方法2 ...]"
    echo "例如: sh run.sh mnist krum fltrust"
    exit 1
fi

DATASET=$1
shift  


for method in "$@"
do
    echo "algo: $method"
	for attack in min-max; do
		for pro in 0.5 0.4 0.3 0.2 0.1; do
    		python main.py --name-dataset $DATASET --agg_method $method --attack_type $attack --mal_prop $pro --alpha 0.8
		done
	done

	# python main.py --name-dataset $DATASET --agg_method $method --attack_type gaussian_noise --mal_prop 0.1 --iid
    # python main.py --name-dataset $DATASET --agg_method $method --attack_type gaussian_noise --mal_prop 0.2 --iid
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type gaussian_noise --mal_prop 0.3 --iid
    # python main.py --name-dataset $DATASET --agg_method $method --attack_type gaussian_noise --mal_prop 0.4 --iid
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type gaussian_noise --mal_prop 0.5 --iid
	
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type sign_flip --mal_prop 0.1 --iid
    # python main.py --name-dataset $DATASET --agg_method $method --attack_type sign_flip --mal_prop 0.2 --iid
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type sign_flip --mal_prop 0.3 --iid
    # python main.py --name-dataset $DATASET --agg_method $method --attack_type sign_flip --mal_prop 0.4 --iid
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type sign_flip --mal_prop 0.5 --iid
	
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type alie --mal_prop 0.1 --iid
    # python main.py --name-dataset $DATASET --agg_method $method --attack_type alie --mal_prop 0.2 --iid
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type alie --mal_prop 0.3 --iid
    # python main.py --name-dataset $DATASET --agg_method $method --attack_type alie --mal_prop 0.4 --iid
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type alie --mal_prop 0.5 --iid
	
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type free_rider --mal_prop 0.1 --iid
    # python main.py --name-dataset $DATASET --agg_method $method --attack_type free_rider --mal_prop 0.2 --iid
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type free_rider --mal_prop 0.3 --iid
    # python main.py --name-dataset $DATASET --agg_method $method --attack_type free_rider --mal_prop 0.4 --iid
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type free_rider --mal_prop 0.5 --iid
	
	# python main.py --name-dataset $DATASET --agg_method $method --alpha 1 
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type label_flip --mal_prop 0.1 --alpha 1
    # python main.py --name-dataset $DATASET --agg_method $method --attack_type label_flip --mal_prop 0.2 --alpha 1
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type label_flip --mal_prop 0.3 --alpha 1
    # python main.py --name-dataset $DATASET --agg_method $method --attack_type label_flip --mal_prop 0.4 --alpha 1
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type label_flip --mal_prop 0.5 --alpha 1
	
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type gaussian_noise --mal_prop 0.1 --alpha 1
    # python main.py --name-dataset $DATASET --agg_method $method --attack_type gaussian_noise --mal_prop 0.2 --alpha 1
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type gaussian_noise --mal_prop 0.3 --alpha 1
    # python main.py --name-dataset $DATASET --agg_method $method --attack_type gaussian_noise --mal_prop 0.4 --alpha 1
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type gaussian_noise --mal_prop 0.5 --alpha 1
	
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type sign_flip --mal_prop 0.1 --alpha 1
    # python main.py --name-dataset $DATASET --agg_method $method --attack_type sign_flip --mal_prop 0.2 --alpha 1
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type sign_flip --mal_prop 0.3 --alpha 1
    # python main.py --name-dataset $DATASET --agg_method $method --attack_type sign_flip --mal_prop 0.4 --alpha 1
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type sign_flip --mal_prop 0.5 --alpha 1
	
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type alie --mal_prop 0.1 --alpha 1
    # python main.py --name-dataset $DATASET --agg_method $method --attack_type alie --mal_prop 0.2 --alpha 1
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type alie --mal_prop 0.3 --alpha 1
    # python main.py --name-dataset $DATASET --agg_method $method --attack_type alie --mal_prop 0.4 --alpha 1
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type alie --mal_prop 0.5 --alpha 1
	
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type free_rider --mal_prop 0.1 --alpha 1
    # python main.py --name-dataset $DATASET --agg_method $method --attack_type free_rider --mal_prop 0.2 --alpha 1
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type free_rider --mal_prop 0.3 --alpha 1
    # python main.py --name-dataset $DATASET --agg_method $method --attack_type free_rider --mal_prop 0.4 --alpha 1
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type free_rider --mal_prop 0.5 --alpha 1
	
	# python main.py --name-dataset $DATASET --agg_method $method --alpha 0.5
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type label_flip --mal_prop 0.1 --alpha 0.5
    # python main.py --name-dataset $DATASET --agg_method $method --attack_type label_flip --mal_prop 0.2 --alpha 0.5
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type label_flip --mal_prop 0.3 --alpha 0.5
    # python main.py --name-dataset $DATASET --agg_method $method --attack_type label_flip --mal_prop 0.4 --alpha 0.5
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type label_flip --mal_prop 0.5 --alpha 0.5
	
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type gaussian_noise --mal_prop 0.1 --alpha 0.5
    # python main.py --name-dataset $DATASET --agg_method $method --attack_type gaussian_noise --mal_prop 0.2 --alpha 0.5
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type gaussian_noise --mal_prop 0.3 --alpha 0.5
    # python main.py --name-dataset $DATASET --agg_method $method --attack_type gaussian_noise --mal_prop 0.4 --alpha 0.5
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type gaussian_noise --mal_prop 0.5 --alpha 0.5
	
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type sign_flip --mal_prop 0.1 --alpha 0.5
    # python main.py --name-dataset $DATASET --agg_method $method --attack_type sign_flip --mal_prop 0.2 --alpha 0.5
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type sign_flip --mal_prop 0.3 --alpha 0.5
    # python main.py --name-dataset $DATASET --agg_method $method --attack_type sign_flip --mal_prop 0.4 --alpha 0.5
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type sign_flip --mal_prop 0.5 --alpha 0.5
	
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type alie --mal_prop 0.1 --alpha 0.5
    # python main.py --name-dataset $DATASET --agg_method $method --attack_type alie --mal_prop 0.2 --alpha 0.5
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type alie --mal_prop 0.3 --alpha 0.5
    # python main.py --name-dataset $DATASET --agg_method $method --attack_type alie --mal_prop 0.4 --alpha 0.5
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type alie --mal_prop 0.5 --alpha 0.5
	
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type free_rider --mal_prop 0.1 --alpha 0.5
    # python main.py --name-dataset $DATASET --agg_method $method --attack_type free_rider --mal_prop 0.2 --alpha 0.5
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type free_rider --mal_prop 0.3 --alpha 0.5
    # python main.py --name-dataset $DATASET --agg_method $method --attack_type free_rider --mal_prop 0.4 --alpha 0.5
	# python main.py --name-dataset $DATASET --agg_method $method --attack_type free_rider --mal_prop 0.5 --alpha 0.5
	
done


echo "Completed all runs."