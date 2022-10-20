if [[ -z $1 ]];
then 
    echo "Download All Results..."
    scp -r nct01141@dt01.bsc.es:/home/nct01/nct01141/.keras/LAB1/result/ ./
else
    while getopts ":v:r:h" optname
    do
        case "$optname" in
        "v")
            ver="$OPTARG"
            ;;
        "r")
            run="$OPTARG"
            ;;
        "h")
            echo "Usage [-v] [-r]"
            ;;
        ":")
            echo "No argument value for option $OPTARG"
            ;;
        "?")
            echo "Unknown option $OPTARG"
            ;;
        *)
            echo "Unknown error while processing options"
            ;;
        esac
        #echo "option index is $OPTIND"
    done

    if [ -z $ver ]; then
        echo "Download All Results..."
        scp -r nct01141@dt01.bsc.es:/home/nct01/nct01141/.keras/LAB1/result/ ./
    elif [ -z $run ]; then 
        echo "Download All runs from version [$ver]"
        scp -r "nct01141@dt01.bsc.es:/home/nct01/nct01141/.keras/LAB1/result/MAMe_v$ver/" "./result/"
    else
        echo "Download version [$ver] run [$run] ..."
        scp -r "nct01141@dt01.bsc.es:/home/nct01/nct01141/.keras/LAB1/result/MAMe_v$ver/run_$run" "./result/MAMe_v$ver/"
    fi
fi