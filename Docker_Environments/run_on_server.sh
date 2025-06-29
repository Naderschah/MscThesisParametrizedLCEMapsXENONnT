SERVER="Server"                       # Replace with your SSH config entry for the server
REMOTE_DIR="/home/felix/Thesis"       # Remote directory on the server
LOCAL_PORT=8888                       # Local port for Jupyter
REMOTE_PORT=8888 

ssh -t -L ${LOCAL_PORT}:localhost:${REMOTE_PORT} ${SERVER} << EOF
  cd ${REMOTE_DIR}
  docker-compose up
EOF

