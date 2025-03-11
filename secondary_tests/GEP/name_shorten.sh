#!/bin/bash

find . -type f -name "*.json" | while read file; do
    dir=$(dirname "$file")
    filename=$(basename "$file")

    newname=$(echo "$filename" | sed 's/.*_//')

    if [ "$filename" != "$newname" ]; then
        mv "$file" "$dir/$newname"
        echo "Renamed: $file -> $dir/$newname"
    fi
done

    find . -type f -name "*.csv" | while read file; do
        dir=$(dirname "$file")
        filename=$(basename "$file")

        newname=$(echo "$filename" | sed -E 's/(.*_)([A-Za-z]+_[0-9]{6,})(.*)_\2(.*)/\1\2\3\4/')

        if [ "$filename" != "$newname" ]; then
            mv "$file" "$dir/$newname"
            echo "Renamed: $file -> $dir/$newname"
        fi
    done