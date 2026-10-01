"""classification / breast_cancer — standalone formula extracted from a trained ML model (numpy only, no model needed).
Usage: predict(**raw_columns) -> prediction ; in_domain(**raw_columns) -> True where inputs are inside the training range."""
import numpy as np
RAW_LO = {'mean_radius': 7.691, 'mean_texture': 9.71, 'mean_smoothness': 0.05263, 'mean_concavity': 0.0, 'mean_concave_points': 0.0, 'worst_radius': 8.678, 'worst_texture': 12.02, 'worst_concave_points': 0.0}
RAW_HI = {'mean_radius': 28.11, 'mean_texture': 39.28, 'mean_smoothness': 0.1634, 'mean_concavity': 0.4268, 'mean_concave_points': 0.2012, 'worst_radius': 36.04, 'worst_texture': 49.54, 'worst_concave_points': 0.2903}
FEAT_LO = {'mean_radius': 7.691, 'mean_texture': 9.71, 'mean_smoothness': 0.05263, 'mean_concavity': 0.0, 'mean_concave_points': 0.0, 'worst_radius': 8.678, 'worst_texture': 12.02, 'worst_concave_points': 0.0}
FEAT_HI = {'mean_radius': 28.11, 'mean_texture': 39.28, 'mean_smoothness': 0.1634, 'mean_concavity': 0.4268, 'mean_concave_points': 0.2012, 'worst_radius': 36.04, 'worst_texture': 49.54, 'worst_concave_points': 0.2903}

def features(mean_radius, mean_texture, mean_smoothness, mean_concavity, mean_concave_points, worst_radius, worst_texture, worst_concave_points):
    """Preprocessing T (raw -> model features)."""
    mean_radius = mean_radius
    mean_texture = mean_texture
    mean_smoothness = mean_smoothness
    mean_concavity = mean_concavity
    mean_concave_points = mean_concave_points
    worst_radius = worst_radius
    worst_texture = worst_texture
    worst_concave_points = worst_concave_points
    return dict(mean_radius=mean_radius, mean_texture=mean_texture, mean_smoothness=mean_smoothness, mean_concavity=mean_concavity, mean_concave_points=mean_concave_points, worst_radius=worst_radius, worst_texture=worst_texture, worst_concave_points=worst_concave_points)

def predict(mean_radius, mean_texture, mean_smoothness, mean_concavity, mean_concave_points, worst_radius, worst_texture, worst_concave_points):
    f = features(mean_radius, mean_texture, mean_smoothness, mean_concavity, mean_concave_points, worst_radius, worst_texture, worst_concave_points); mean_radius = f['mean_radius']; mean_texture = f['mean_texture']; mean_smoothness = f['mean_smoothness']; mean_concavity = f['mean_concavity']; mean_concave_points = f['mean_concave_points']; worst_radius = f['worst_radius']; worst_texture = f['worst_texture']; worst_concave_points = f['worst_concave_points']
    return 1 / (1 + np.exp(-(22.836*mean_concave_points*mean_concavity + 0.54365*mean_concave_points*mean_radius - 77.915*mean_concave_points*mean_smoothness + 0.52544*mean_concave_points*mean_texture + 0.98576*mean_concave_points*worst_radius - 59.882*mean_concave_points + 13.656*mean_concavity**2 + 0.2566*mean_concavity*mean_radius - 44.715*mean_concavity*mean_smoothness + 9.5786*mean_concavity*worst_concave_points + 0.44727*mean_concavity*worst_radius - 0.097121*mean_concavity*worst_texture - 15.13*mean_concavity - 0.003763*mean_radius**2 - 0.0020231*mean_radius*worst_texture - 0.23816*mean_radius + 167.4*mean_smoothness**2 - 55.774*mean_smoothness*worst_concave_points - 1.7225*mean_smoothness*worst_texture + 0.0039306*mean_texture**2 + 0.00609*mean_texture*worst_radius + 0.0078864*mean_texture*worst_texture - 0.52012*mean_texture - 28.29*worst_concave_points**2 + 0.22851*worst_concave_points*worst_radius - 0.2335*worst_concave_points*worst_texture - 8.8393*worst_concave_points - 0.78816*worst_radius + 0.0032076*worst_texture**2 - 0.37576*worst_texture + 35.7605)))

def in_domain(mean_radius, mean_texture, mean_smoothness, mean_concavity, mean_concave_points, worst_radius, worst_texture, worst_concave_points):
    raw = dict(mean_radius=mean_radius, mean_texture=mean_texture, mean_smoothness=mean_smoothness, mean_concavity=mean_concavity, mean_concave_points=mean_concave_points, worst_radius=worst_radius, worst_texture=worst_texture, worst_concave_points=worst_concave_points); f = features(**raw); ok = True
    for k, v in raw.items(): ok = ok & (np.asarray(v) >= RAW_LO[k]) & (np.asarray(v) <= RAW_HI[k])
    for k, v in f.items(): ok = ok & (np.asarray(v) >= FEAT_LO[k]) & (np.asarray(v) <= FEAT_HI[k])
    return ok
