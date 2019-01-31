#!/bin/bash

# extract different reliability genelist form subcelluar_location.csv


input=subcellular_location.tsv
enhanced=enhanced.list
supported=supported.list
approved=approved.list
uncertain=uncertain.list

awk '$3 == "Enhanced" {print $1}' $input | sort | uniq > $enhanced
awk '$3 == "Supported" {print $1}' $input | sort | uniq > $supported
awk '$3 == "Approved" {print $1}' $input | sort | uniq > $approved
awk '$3 == "Uncertain" {print $1}' $input | sort | uniq > $uncertain
