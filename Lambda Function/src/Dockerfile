FROM public.ecr.aws/lambda/provided:al2023

WORKDIR /tmp
RUN dnf install -y python3

# Copy custom runtime bootstrap
COPY bootstrap ${LAMBDA_RUNTIME_DIR}
RUN chmod +x ${LAMBDA_RUNTIME_DIR}/bootstrap

# requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN pip3 install -r ${LAMBDA_TASK_ROOT}/requirements.txt -t ${LAMBDA_RUNTIME_DIR}/ 

# Copy function code
COPY handler.py ${LAMBDA_TASK_ROOT}/handler.py

CMD ["handler.lambda_handler"]