# collect commit dates (author date) as YYYY-MM-DD in Asia/Kolkata
          mapfile -t DATES < <(git log --date=format:%Y-%m-%d --pretty=%ad)
          # put into an associative set
          declare -A HAS
          for d in "${DATES[@]}"; do HAS["$d"]=1; done

          STREAK=0
          i=0
          while true; do
            DAY=$(date -d "$TODAY - $i day" +%Y-%m-%d)
            if [[ -n ${HAS[$DAY]} ]]; then
              STREAK=$((STREAK + 1))
              i=$((i + 1))
            else
              break
            fi
          done

          echo "streak=$STREAK" 
