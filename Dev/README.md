# Dockerfile for development and testing

## `Dockerfile_bash`
`Dockerfile_bash` uses personal configurations to create a development workstation with project based `requirements.txt` 

### Build commands
Build from project directory (parent directory), fixes a lot of issues with not being able to access `requirements.txt`

`docker build -t ${project_name}:dev -f Dev/Dockerfile_bash .` 

`docker run --rm -it -v "$(pwd)":/workdir ${project_name}:dev`

## `Dockerfile_jupyter`

`Dockerfile_jupyter` is created only for `requirements.txt` implementation, no major additional changes have been made.

### Build commands

Build from project directory (parent directory), fixes a lot of issues with not being able to access `requirements.txt`

`docker build -t ${project_name}:jupyter -f Dev/Dockerfile_jupyter .`

`docker run --rm -d --network host -v "$(pwd)":/workdir ${project_name}:jupyter`
