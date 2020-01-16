
ARG REGION
FROM 763104351884.dkr.ecr.$REGION.amazonaws.com/mxnet-inference:1.6.0-gpu-py3

RUN pip install --upgrade --user --pre 'mxnet-mkl' 'https://github.com/dmlc/gluon-nlp/tarball/v0.9.x'
RUN pip list | grep mxnet