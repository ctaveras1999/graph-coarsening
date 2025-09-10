for i in 0 1 2 3 4 5 6; do 
    for j in 0 1 2 3 4 5; do 
        nohup python graph_clustering.py $i $j &  
    done 
done