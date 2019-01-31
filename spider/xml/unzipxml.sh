#!/bin/bash

for f in `ls *.gz`
do
    gunzip $f
done
