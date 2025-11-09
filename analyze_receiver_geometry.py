#!/usr/bin/env python3
"""
Analyze WebSDR receiver geometry to understand why GDOP is so high.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform

# WebSDR receiver positions (lat, lon in degrees)
receivers = {
    "Aquila di Giaveno": (45.03, 7.27),
    "Coazze": (45.03, 7.27),  # DUPLICATE position!
    "Genova": (44.395, 8.956),
    "Milano - Baggio": (45.478, 9.123),
    "Montanaro": (45.234, 7.857),
    "Passo del Giovi": (44.561, 8.956),
    "Torino": (45.044, 7.672),
}

def lat_lon_to_xy(lat, lon):
    """Simple equirectangular projection (good enough for small regions)."""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # Center around mean position
    lat_mean = 44.820
    lon_mean = 8.12
    
    R = 6371.0  # Earth radius in km
    x = R * (lon_rad - np.radians(lon_mean)) * np.cos(np.radians(lat_mean))
    y = R * (lat_rad - np.radians(lat_mean))
    
    return x, y

print("=" * 70)
print("WebSDR RECEIVER GEOMETRY ANALYSIS")
print("=" * 70)
print()

# Convert to XY coordinates
positions = []
for name, (lat, lon) in receivers.items():
    x, y = lat_lon_to_xy(lat, lon)
    positions.append((name, x, y, lat, lon))
    print(f"{name:20} Lat: {lat:7.3f}° Lon: {lon:7.3f}° → X: {x:7.1f} km Y: {y:7.1f} km")

print()
print("-" * 70)
print("GEOMETRY ISSUES FOUND:")
print("-" * 70)
print()

# Check for duplicate positions
coords = [(x, y) for _, x, y, _, _ in positions]
coords_array = np.array(coords)

print("1. DUPLICATE POSITIONS:")
for i in range(len(positions)):
    for j in range(i + 1, len(positions)):
        dist = np.linalg.norm(coords_array[i] - coords_array[j])
        if dist < 1.0:  # Less than 1 km apart
            print(f"   ⚠️  {positions[i][0]:20} and {positions[j][0]:20} → {dist:.2f} km apart (DUPLICATE!)")

print()
print("2. PAIRWISE DISTANCES:")
distances = pdist(coords_array)
dist_matrix = squareform(distances)

for i in range(len(positions)):
    for j in range(i + 1, len(positions)):
        dist_km = dist_matrix[i, j]
        print(f"   {positions[i][0]:20} ↔ {positions[j][0]:20} → {dist_km:6.1f} km")

print()
print(f"   Min distance: {distances.min():.1f} km")
print(f"   Max distance: {distances.max():.1f} km")
print(f"   Mean distance: {distances.mean():.1f} km")
print(f"   Median distance: {np.median(distances):.1f} km")

print()
print("3. CLUSTER ANALYSIS:")
print(f"   Centroid: {coords_array.mean(axis=0)}")
print(f"   Std dev: X={coords_array[:, 0].std():.1f} km, Y={coords_array[:, 1].std():.1f} km")

# Check for collinearity (all receivers in a line)
print()
print("4. COLLINEARITY CHECK:")
# Perform PCA to find principal axes
mean_pos = coords_array.mean(axis=0)
centered = coords_array - mean_pos
cov_matrix = np.cov(centered.T)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

variance_ratio = eigenvalues[0] / eigenvalues[1]
print(f"   Variance ratio (PC1/PC2): {variance_ratio:.2f}")
if variance_ratio > 10:
    print(f"   ⚠️  HIGHLY COLLINEAR (ratio > 10)! Receivers form a line.")
elif variance_ratio > 5:
    print(f"   ⚠️  MODERATELY COLLINEAR (ratio > 5). Poor triangulation geometry.")
else:
    print(f"   ✅ Good 2D spread (ratio < 5)")

print()
print("5. AREA COVERAGE:")
# Calculate convex hull area (rough estimate)
from scipy.spatial import ConvexHull
hull = ConvexHull(coords_array)
area_km2 = hull.volume  # 'volume' is area in 2D
print(f"   Convex hull area: {area_km2:.1f} km²")
print(f"   Perimeter: {hull.area:.1f} km")  # 'area' is perimeter in 2D

print()
print("=" * 70)
print("GDOP IMPACT ANALYSIS")
print("=" * 70)
print()

print("Expected GDOP characteristics:")
print()
print("  With 7 receivers:")
print("  - Good geometry (spread, non-collinear): GDOP < 5")
print("  - Poor geometry (clustered, collinear): GDOP 10-50")
print("  - Very poor geometry (duplicates, line): GDOP > 50")
print()
print("  Observed GDOP: 111.5 (EXTREMELY POOR)")
print()

print("Contributing factors to high GDOP:")
factors = []

# Check duplicates
num_duplicates = np.sum(distances < 1.0)
if num_duplicates > 0:
    factors.append(f"• {num_duplicates} duplicate receiver position(s)")
    print(f"  ⚠️  {num_duplicates} duplicate receiver position(s)")

# Check clustering
if distances.mean() < 50:
    factors.append(f"• Small baseline (mean distance {distances.mean():.1f} km)")
    print(f"  ⚠️  Small baseline (mean distance {distances.mean():.1f} km)")

# Check collinearity
if variance_ratio > 5:
    factors.append(f"• Receivers are collinear (variance ratio {variance_ratio:.1f})")
    print(f"  ⚠️  Receivers are collinear (variance ratio {variance_ratio:.1f})")

print()
print("RECOMMENDATIONS:")
print()
print("1. Remove duplicate receiver: Coazze (same position as Aquila di Giaveno)")
print("2. Add receivers outside the Turin-Genoa-Milan triangle")
print("3. Target receiver spacing: 100-200 km for good GDOP")
print("4. Add receivers with diverse azimuths from typical target area")
print()
print("Expected improvement:")
print("  - Removing duplicate: GDOP 111 → ~80 (still poor)")
print("  - Adding 2-3 geographically diverse receivers: GDOP < 10 (good)")
print()

print("=" * 70)
