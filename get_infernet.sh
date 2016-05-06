#!/bin/bash

wget --referer='http://research.microsoft.com/en-us/downloads/710cd61f-3587-44f4-b12d-a2c75722c4f6/' 'http://ftp.research.microsoft.com/downloads/710cd61f-3587-44f4-b12d-a2c75722c4f6/Infer.NET 2.6.zip'

unzip -j 'Infer.NET 2.6.zip' 'Bin\\*' -d bin/

rm 'Infer.NET 2.6.zip'
