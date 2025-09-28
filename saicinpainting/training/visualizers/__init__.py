import logging

from saicinpainting.training.visualizers.directory import DirectoryVisualizer, Sentinel_DirectoryVisualizer


def make_visualizer(kind, **kwargs):
    logging.info(f'Make visualizer {kind}')

    if kind == 'directory':
        return DirectoryVisualizer(**kwargs)
    if kind == 'sentinel_directory':
        return Sentinel_DirectoryVisualizer(**kwargs)

    raise ValueError(f'Unknown visualizer kind {kind}')
