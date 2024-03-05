#!/bin/sh

while [ ! -f ./USPEX_IS_DONE ]; do
   date >> log 
   octave -W < USPEX.m  >> log
   sleep 60
done
