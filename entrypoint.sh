#!/bin/bash

# Increase the inotify limit for file monitoring
echo fs.inotify.max_user_watches=524288 | tee -a /etc/sysctl.conf && sysctl -p

# Start the Streamlit app
exec streamlit run app.py
