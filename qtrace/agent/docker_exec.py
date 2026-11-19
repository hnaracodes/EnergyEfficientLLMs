"""A minimal container handle: `docker exec` through the CLI, no SDK dependency.

Used by the grader and the fingerprint helper. Commands run in a separate exec, never through the
model's shell, so nothing they print reaches the model's context.
"""

import subprocess

# PATH used by R2E-Gym's own runtime (r2egym/agenthub/runtime/docker.py), so tools resolve the same way.
R2E_PATH = "/root/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


class DockerContainer:
    def __init__(self, name, docker="docker"):
        self.name = name
        self.docker = docker

    def exec(self, script, workdir="/testbed", timeout=300, env_path=R2E_PATH, stdin=None):
        """Run a shell script in the container. Returns (stdout + stderr as bytes, exit code).

        Exit code 124 means the in-container `timeout` fired. `stdin` (bytes) is piped in.
        """
        cmd = [self.docker, "exec", *(["-i"] if stdin is not None else []), "-w", workdir, "-e",
               f"PATH={env_path}", self.name, "timeout", str(int(timeout)), "/bin/sh", "-c", script]
        proc = subprocess.run(cmd, input=stdin, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              timeout=timeout + 30)
        return proc.stdout, proc.returncode

    @classmethod
    def start(cls, image, name, docker="docker", platform="linux/amd64"):
        """Start a long-lived container from a task image (tests use this; the harness has its own)."""
        subprocess.run([docker, "run", "-d", "--rm", "--platform", platform, "--network", "none", "--name", name,
                        image, "sleep", "infinity"], check=True, stdout=subprocess.DEVNULL)
        return cls(name, docker)

    def stop(self):
        subprocess.run([self.docker, "rm", "-f", self.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
