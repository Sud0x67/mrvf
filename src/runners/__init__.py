REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner
from .parallel_runner_maven import ParallelRunnerMaven
REGISTRY["parallel_maven"] = ParallelRunnerMaven
