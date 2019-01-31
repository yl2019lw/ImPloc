#!/bin/bash

# generate label set for 2000 gene with supported reliability

input=supported_2000.txt
output=slabel.txt

awk -F',' 'NR > 1 {print $2}' $input > tmp.txt
sed 's/;/\n/g' tmp.txt | sed '/^$/d' | sort | uniq > $output
rm tmp.txt
