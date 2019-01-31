#!/bin/bash

# use md5sum to check crawled image file integrity

md5check=md5sum.txt
if [ -e $md5check ]; then
    rm $md5check
fi

dir=("/data/longwei/hpa/data", "/ndata/longwei/hpa/data")


for datadir in ${dir[@]};
do
    for f in `find $datadir -type f`;
    do
        while [ $(ps aux |grep md5sum | wc -l) -ge 100 ]
        do
            echo "------sleep-------"
            sleep 1
        done
        {
            echo "--------check $f--------------"
            md5sum $f >> $md5check
            exit
        }&
    done
done

wait
