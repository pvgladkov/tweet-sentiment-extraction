FROM gcr.io/kaggle-gpu-images/python

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN echo $PROJECT_ROOT
COPY ./requirements.txt $PROJECT_ROOT/

RUN pip install --upgrade pip
#RUN pip install --no-cache-dir -r requirements.txt

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

ENV PYTHONPATH "${PYTHONPATH}:${PROJECT_ROOT}"

CMD ["/bin/bash"]