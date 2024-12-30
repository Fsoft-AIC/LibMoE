# if you get "not permission file" then:
export TMUX_TMPDIR=~/tmux_tmp
# write terminate into to file .txt
tmux capture-pane -pS - > ./assets/result_eval/log.txt
# watch hidden progress of the tmux
tmux a
