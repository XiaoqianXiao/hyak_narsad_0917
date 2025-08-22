#!/bin/bash

# Print filename if last line does NOT contain "Group-level analysis pipeline completed successfully"

for file in *.err; do
    if [[ -f "$file" ]]; then
        last_line=$(tail -n 1 "$file" 2>/dev/null)
        if [[ "$last_line" != *"analysis pipeline completed successfully"* ]]; then
            echo "$file"
        fi
    fi
done
