# Encryption showcase

Please, do **NOT** use the provided encryption keys in production! Those are added only for convenience.

## Running without encryption

1. Run control program in master node:

    ```bash
    uniflex-agent --config ./config_master.yaml
    ```

2. Run modules in slave node:

    ```bash
    uniflex-agent --config ./config_slave.yaml
    ```

## Running with encryption

1. Run control program in master node:

    ```bash
    uniflex-agent --config ./encryption_master.yaml
    ```

2. Run modules in slave node:

    ```bash
    uniflex-agent --config ./encryption_slave.yaml
    ```


For debugging mode run with `-v` option
