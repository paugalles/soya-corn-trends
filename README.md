# Commodities

This is an exercice of analyzing commodity trading data, understanding it and forecasting.

There are two notebooks:

* `Visualization.ipynb` > This is the visualization of the data.
* `MODEL.ipynb` > This is a model made from this data.

## Install

Create a python environment, then install the requirements with:

```bash
pip3 install -r requirements.txt
```

Alternatively you can use docker and nvidia-docker as follows:

1. Build the docker image

```bash
make build
```

2. Raise a running container

```bash
make container
```

3. As soon as the docker is running in background you can launch the following services whithin it:

    * Launch a jupyter lab server

        ```bash
        make nb
        ```

        Then access it from your browser by using this address `localhost:8088/?token=commodities`

    * Stops the jupyter server

        ```bash
        make nbstop
        ```

    * Launch an mlflow server

        ```bash
        make mlf
        ```
        Then access it from your browser by using this address `localhost:5055`

    * To raise an interactive shell from our running container

        ```bash
        make execsh
        ```
        
    * To run tests

        ```bash
        make test
        ```

    * Stop the docker container and everything that is running within it

        ```bash
        make stop
        ```