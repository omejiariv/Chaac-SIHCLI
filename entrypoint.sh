#!/bin/bash

# Aumenta el límite de inotify para la monitorización de archivos
echo fs.inotify.max_user_watches=524288 | tee -a /etc/sysctl.conf && sysctl -p

# Inicia la aplicación de Streamlit
exec streamlit run app.py