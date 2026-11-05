import numpy as np
import pandas as pd
import xarray as xr
import gsw
from pyproj import Geod
from pathlib import Path

# ============================================================
# PARAMETERS
# ============================================================

BATHY_FILE = "/media/hargous/MyPassport/OT_Ibiza/data/Bathy_Ibiza_channel/bathy_westmed.nc"

INPUT_DIR = "/media/hargous/MyPassport/OT_Ibiza/data/SOCIB/canalesL2/"
OUTPUT_DIR = "/media/hargous/MyPassport/OT_Ibiza/results/vgeo/"

LAT_SECTION = 39.0
LON_MIN = 0.1
LON_MAX = 1.13
MAX_DISTANCE_KM = 15.0

DX = 2.0  # horizontal resolution [km]
DZ = 5.0  # vertical resolution [m]
Z_MAX = 1000.0  # maximum depth [m]

FILTER_LENGTH = 24.0  # horizontal smoothing length [km]

MIN_REVERSE_DISTANCE_DEG = 0.10

# ============================================================
# BATHYMETRY
# ============================================================


def load_bathymetry():

    bathy = xr.open_dataset(BATHY_FILE)

    bathy_39 = bathy.sel(lon=slice(LON_MIN, LON_MAX)).sel(lat=LAT_SECTION, method="nearest")

    # Latitude of the section (for distance calculation)
    lat_section = float(bathy_39.lat)

    # Longitudes
    lon = bathy_39.lon.values
    n = len(lon)

    # Géodésique WGS84
    geod = Geod(ellps="WGS84")

    # Distance from LON_MIN
    _, _, x_m = geod.inv(np.full(n, LON_MIN), np.full(n, lat_section), lon, np.full(n, lat_section))

    x_km = x_m / 1000.0

    # Add x as a coordinate to the bathymetry dataset
    bathy_39 = bathy_39.assign_coords(x=("lon", x_km))

    return bathy_39


# ============================================================
# EXTRACT TRAJECTORY AND SECTION
# ============================================================


def extract_section(ds):

    geod = Geod(ellps="WGS84")

    time = ds.time.values
    lat = ds.latitude.values
    lon = ds.longitude.values

    n = len(time)

    print(f"Number of profiles: {n}")

    # --------------------------------------------------------
    # Distance between each profile and the section
    # --------------------------------------------------------

    _, _, distance_m = geod.inv(lon, lat, lon, np.full(n, LAT_SECTION))

    distance_km = distance_m / 1000.0

    within_15km = distance_km <= MAX_DISTANCE_KM

    # --------------------------------------------------------
    # Distance along the section from LON_MIN
    # --------------------------------------------------------

    _, _, x_m = geod.inv(np.full(n, LON_MIN), np.full(n, LAT_SECTION), lon, np.full(n, LAT_SECTION))

    x_km = x_m / 1000.0

    # --------------------------------------------------------
    # Length of the section
    # --------------------------------------------------------

    _, _, section_length_m = geod.inv(LON_MIN, LAT_SECTION, LON_MAX, LAT_SECTION)

    section_length_km = section_length_m / 1000.0

    # --------------------------------------------------------
    # Detection of direction (Eastward or Westward)
    # --------------------------------------------------------

    dlon = np.diff(lon)

    lon_threshold = 0.0

    direction_step = np.zeros(n - 1)

    direction_step[dlon > lon_threshold] = 1
    direction_step[dlon < -lon_threshold] = -1

    # ------------------------------------------------------------
    # Cleaning of direction to remove small reversals
    # ------------------------------------------------------------

    direction_clean = np.full(n, np.nan)

    # Find the first valid direction (not zero)
    valid_direction = np.where(direction_step != 0)[0]

    if len(valid_direction) == 0:

        direction_clean[:] = np.nan

    else:

        first = valid_direction[0]
        direction_clean[: first + 1] = direction_step[first]

        current_direction = direction_step[first]

        i = first + 1

        while i < len(direction_step):

            step_direction = direction_step[i]

            # Same direction as before
            if step_direction == current_direction:

                direction_clean[i + 1] = current_direction
                i += 1
                continue

            # No direction (zero) - keep the previous direction
            if step_direction == 0:

                direction_clean[i + 1] = current_direction
                i += 1
                continue

            # ----------------------------------------------------
            # Change of direction detected
            # ----------------------------------------------------
            # If the direction changes, we need to check if it's a real change or just a small reversal.

            reverse_direction = step_direction

            j = i
            reverse_distance = 0.0

            while j < len(direction_step) and direction_step[j] == reverse_direction:

                reverse_distance += abs(dlon[j])
                j += 1

            # ----------------------------------------------------
            # Small reversal - keep the previous direction
            # ----------------------------------------------------

            if reverse_distance < MIN_REVERSE_DISTANCE_DEG:

                direction_clean[i + 1 : j + 1] = current_direction

                i = j

            # ----------------------------------------------------
            # Real change of direction - update the current direction
            # ----------------------------------------------------

            else:

                current_direction = reverse_direction

                direction_clean[i + 1 : j + 1] = current_direction

                i = j

    # NaN values at the beginning or end of the array are filled with the nearest valid direction
    direction_clean = pd.Series(direction_clean).ffill().bfill().values

    direction = direction_clean

    # --------------------------------------------------------
    # Leg detection based on direction changes
    # --------------------------------------------------------

    new_leg = np.zeros(n, dtype=bool)

    new_leg[0] = True

    new_leg[1:] = direction[1:] != direction[:-1]

    leg = np.cumsum(new_leg)

    print(f"Number of legs detected: " f"{leg.max()}")

    # --------------------------------------------------------
    # Trajectory dataframe
    # --------------------------------------------------------

    trajectory = pd.DataFrame(
        {
            "profile_index": ds.profile_index.values,
            "time": time,
            "latitude": lat,
            "longitude": lon,
            "distance_km": distance_km,
            "x_km": x_km,
            "direction": direction,
            "leg": leg,
            "within_15km": within_15km,
        }
    )

    # --------------------------------------------------------
    # Selection of profiles within the section and within the distance limit
    # --------------------------------------------------------

    valid = (
        (trajectory.distance_km <= MAX_DISTANCE_KM)
        & (trajectory.longitude >= LON_MIN)
        & (trajectory.longitude <= LON_MAX)
    )

    trajectory_section = trajectory.loc[valid].copy()

    print(f"Profiles retained: " f"{len(trajectory_section)} / " f"{len(trajectory)}")

    for leg_number in np.unique(trajectory_section["leg"]):

        leg_data = trajectory_section[trajectory_section["leg"] == leg_number]

        if direction[int(leg_data.index[0])] > 0:
            direction_name = "EAST"
        else:
            direction_name = "WEST"

        print(
            f"Leg {leg_number}: "
            f"{leg_data.longitude.iloc[0]:.3f} → "
            f"{leg_data.longitude.iloc[-1]:.3f} °E, "
            f"{len(leg_data)} profiles, "
            f"{direction_name}"
        )

    return (trajectory_section, section_length_km)


# ============================================================
# PROCESS ONE LEG
# ============================================================


def process_leg(ds_leg, x_std, z_std, bathy_39, section_length_km):

    # ========================================================
    # 1. Prepare data
    # ========================================================

    # Variable names may vary depending on the glider type (e.g., Slocum, Seaglider, etc.)
    salinity_var = (
        "salinity_corrected_thermal" if "salinity_corrected_thermal" in ds_leg else "salinity"
    )

    density_var = (
        "density_corrected_thermal" if "density_corrected_thermal" in ds_leg else "density"
    )

    df = (
        ds_leg[["temperature", salinity_var, density_var, "depth", "x_km", "profile_index"]]
        .to_dataframe()
        .reset_index()
    )

    # ========================================================
    # 2. Binning
    # ========================================================

    z_bins = np.arange(0, Z_MAX + DZ, DZ)

    x_bins = np.arange(0, section_length_km + DX, DX)

    z_centers = (z_bins[:-1] + z_bins[1:]) / 2

    x_centers = (x_bins[:-1] + x_bins[1:]) / 2

    df["x_bin"] = pd.cut(df["x_km"], bins=x_bins, labels=x_centers, include_lowest=True)

    df["z_bin"] = pd.cut(df["depth"], bins=z_bins, labels=z_centers, include_lowest=True)

    df = df.dropna(subset=["x_bin", "z_bin"])

    # ========================================================
    # 3. Mean in each bin
    # ========================================================

    binned = (
        df.groupby(["x_bin", "z_bin"], observed=True)[["temperature", salinity_var, density_var]]
        .mean()
        .reset_index()
    )

    binned["x_bin"] = binned["x_bin"].astype(float)

    binned["z_bin"] = binned["z_bin"].astype(float)

    # ========================================================
    # 4. 2D fields
    # ========================================================

    theta_binned = binned.pivot(index="z_bin", columns="x_bin", values="temperature")

    salinity_binned = binned.pivot(index="z_bin", columns="x_bin", values=salinity_var)

    density_binned = binned.pivot(index="z_bin", columns="x_bin", values=density_var)

    # ========================================================
    # 5. Maximum observed depth
    # ========================================================

    x_binned = theta_binned.columns.values.astype(float)

    z_binned = theta_binned.index.values.astype(float)

    depth_max_x = np.full(len(x_binned), np.nan)

    for i, x in enumerate(x_binned):

        valid = np.isfinite(theta_binned[x].values)

        if np.any(valid):

            depth_max_x[i] = z_binned[valid].max()

    # ========================================================
    # 6. Horizontal interpolation
    # ========================================================

    def interpolate_horizontal(dataframe, x_standard):

        x = dataframe.columns.values.astype(float)

        z = dataframe.index.values.astype(float)

        data = dataframe.values.astype(float)

        result = np.full((len(z), len(x_standard)), np.nan)

        for iz in range(len(z)):

            values = data[iz, :]

            valid = np.isfinite(values)

            if valid.sum() >= 2:

                result[iz, :] = np.interp(
                    x_standard, x[valid], values[valid], left=np.nan, right=np.nan
                )

        return pd.DataFrame(result, index=z, columns=x_standard)

    theta_interp = interpolate_horizontal(theta_binned, x_std)

    salinity_interp = interpolate_horizontal(salinity_binned, x_std)

    density_interp = interpolate_horizontal(density_binned, x_std)

    # ========================================================
    # 7. Maximum observed depth
    # ========================================================

    valid_depth = np.isfinite(depth_max_x)

    depth_max_std = np.interp(
        x_std, x_binned[valid_depth], depth_max_x[valid_depth], left=np.nan, right=np.nan
    )

    # ========================================================
    # 8. Bathymetry
    # ========================================================

    bathy_x = bathy_39.x.values

    bathy_depth = -bathy_39.elevation.values

    bathy_std = np.interp(x_std, bathy_x, bathy_depth, left=np.nan, right=np.nan)

    depth_limit = np.minimum(depth_max_std, bathy_std)

    # ========================================================
    # 9. Standard vertical grid
    # ========================================================

    theta_std = theta_interp.reindex(z_std)

    salinity_std = salinity_interp.reindex(z_std)

    density_std = density_interp.reindex(z_std)

    # ========================================================
    # 10. Mask below depth limit
    # ========================================================

    for ix, x in enumerate(x_std):

        if np.isfinite(depth_limit[ix]):

            mask = z_std > depth_limit[ix]

            theta_std.iloc[mask, ix] = np.nan

            salinity_std.iloc[mask, ix] = np.nan

            density_std.iloc[mask, ix] = np.nan

    # ========================================================
    # 11. Final dataset
    # ========================================================

    ds_standard = xr.Dataset(
        {
            "temperature": (("depth", "x"), theta_std.values),
            "salinity": (("depth", "x"), salinity_std.values),
            "density": (("depth", "x"), density_std.values),
            "depth_max": (("x",), depth_max_std),
            "bathymetry": (("x",), bathy_std),
            "depth_limit": (("x",), depth_limit),
        },
        coords={"depth": z_std, "x": x_std},
    )

    # ========================================================
    # Attributes
    # ========================================================

    ds_standard["temperature"].attrs = {"long_name": "sea_water_temperature", "units": "degC"}

    ds_standard["salinity"].attrs = {
        "long_name": "water salinity from raw conductivity "
        "and temperature with thermal lag corrected",
        "units": "psu",
    }

    ds_standard["density"].attrs = {
        "long_name": "water density using salinity from raw "
        "conductivity and temperature with "
        "thermal lag corrected",
        "units": "kg m-3",
    }

    ds_standard["depth_max"].attrs = {"long_name": "Maximum observed depth", "units": "m"}

    ds_standard["bathymetry"].attrs = {"long_name": "Bathymetric depth", "units": "m"}

    ds_standard["depth_limit"].attrs = {
        "long_name": "Minimum of maximum observed depth " "and bathymetric depth",
        "units": "m",
    }

    ds_standard.attrs["horizontal_resolution"] = DX

    ds_standard.attrs["vertical_resolution"] = DZ

    ds_standard.attrs["description"] = (
        "Glider section binned at 2 km x 5 m "
        "and horizontally interpolated onto "
        "a standard grid. No extrapolation "
        "below the maximum observed depth."
    )

    return ds_standard


# ============================================================
# BUILD ALL LEGS
# ============================================================


def build_legs(ds, bathy_39, trajectory_section, section_length_km):

    # --------------------------------------------------------
    # Selected profiles
    # --------------------------------------------------------

    profile_indices = trajectory_section["profile_index"].values

    ds_section = ds.where(ds.profile_index.isin(profile_indices), drop=True)

    # --------------------------------------------------------
    # Associate each profile with its leg
    # --------------------------------------------------------

    leg_section = trajectory_section.set_index("profile_index")["leg"]

    ds_section = ds_section.assign_coords(
        leg=("time", leg_section.loc[ds_section.profile_index.values].values)
    )

    # --------------------------------------------------------
    # Associate x position with each profile
    # --------------------------------------------------------

    x_section = trajectory_section.set_index("profile_index")["x_km"]

    ds_section = ds_section.assign_coords(
        x_km=("time", x_section.loc[ds_section.profile_index.values].values)
    )

    # --------------------------------------------------------
    # Standard grid
    # --------------------------------------------------------

    x_std = np.arange(DX / 2, section_length_km, DX)

    z_std = np.arange(DZ / 2, Z_MAX, DZ)

    ds_legs = {}

    # --------------------------------------------------------
    # Loop over legs
    # --------------------------------------------------------
    skipped_legs = 0
    for leg_number in sorted(trajectory_section["leg"].unique()):

        trajectory_leg = trajectory_section[trajectory_section["leg"] == leg_number]

        # ----------------------------------------------------
        # Remove legs located entirely in the central section
        # ----------------------------------------------------

        lon_min_leg = trajectory_leg["longitude"].min()
        lon_max_leg = trajectory_leg["longitude"].max()

        if (lon_min_leg > 0.4) or (lon_max_leg < 0.9):

            print(f"SKIPPING LEG {leg_number}: " f"{lon_min_leg:.3f} → " f"{lon_max_leg:.3f} °E")

            skipped_legs += 1
            continue

        profiles_leg = trajectory_leg["profile_index"].values

        # ----------------------------------------------------
        # Time at middle of leg
        # ----------------------------------------------------

        time_start = trajectory_leg["time"].iloc[0]

        time_end = trajectory_leg["time"].iloc[-1]

        time_leg = time_start + (time_end - time_start) / 2

        # ----------------------------------------------------
        # Dataset corresponding to the leg
        # ----------------------------------------------------

        ds_leg = ds_section.where(ds_section.profile_index.isin(profiles_leg), drop=True)

        # ----------------------------------------------------
        # Process / interpolate
        # ----------------------------------------------------

        ds_standard_leg = process_leg(
            ds_leg, x_std=x_std, z_std=z_std, bathy_39=bathy_39, section_length_km=section_length_km
        )

        # ----------------------------------------------------
        # Longitude on standard grid
        # ----------------------------------------------------

        x_leg = trajectory_leg["x_km"].values

        lon_leg = trajectory_leg["longitude"].values

        # np.interp requires increasing x
        if x_leg[0] > x_leg[-1]:

            x_leg = x_leg[::-1]
            lon_leg = lon_leg[::-1]

        lon_std = np.interp(x_std, x_leg, lon_leg)

        # ----------------------------------------------------
        # Add coordinates
        # ----------------------------------------------------

        ds_standard_leg = ds_standard_leg.assign_coords(time=time_leg, lon=("x", lon_std))

        # ----------------------------------------------------
        # Attributes
        # ----------------------------------------------------

        ds_standard_leg.attrs["leg"] = int(leg_number)

        direction_leg = trajectory_leg["direction"].iloc[0]

        ds_standard_leg.attrs["direction"] = "Eastward" if direction_leg > 0 else "Westward"

        ds_legs[int(leg_number)] = ds_standard_leg

    print(f"LEGS SKIPPED: {skipped_legs}")

    return ds_legs


# ============================================================
# SMOOTH ALL LEGS
# ============================================================


def smooth_legs(ds_legs):

    ds_legs_smooth = {}

    window = int(FILTER_LENGTH / DX)

    for leg_number, ds_leg in ds_legs.items():

        # ----------------------------------------------------
        # Horizontal smoothing
        # ----------------------------------------------------

        temperature_smooth = (
            ds_leg["temperature"].rolling(x=window, center=True, min_periods=1).mean()
        )

        salinity_smooth = ds_leg["salinity"].rolling(x=window, center=True, min_periods=1).mean()

        # ----------------------------------------------------
        # Bathymetric mask
        # ----------------------------------------------------

        mask_bathy = ds_leg["depth"] <= ds_leg["bathymetry"]

        temperature_smooth = temperature_smooth.where(mask_bathy)

        salinity_smooth = salinity_smooth.where(mask_bathy)

        # ----------------------------------------------------
        # Add variables
        # ----------------------------------------------------

        ds_leg = ds_leg.copy()

        ds_leg["temperature_smooth"] = temperature_smooth

        ds_leg["salinity_smooth"] = salinity_smooth

        ds_leg["temperature_smooth"].attrs = {
            "long_name": "Horizontally smoothed " "sea water temperature",
            "units": "degC",
            "smoothing_filter": "moving average",
            "filter_length": "24 km",
        }

        ds_leg["salinity_smooth"].attrs = {
            "long_name": "Horizontally smoothed salinity",
            "units": "psu",
            "smoothing_filter": "moving average",
            "filter_length": "24 km",
        }

        ds_legs_smooth[leg_number] = ds_leg

    return ds_legs_smooth


# ============================================================
# GEOSTROPHIC VELOCITY
# ============================================================


def compute_geostrophic_velocity(ds, lat=39.0):

    g = 9.81
    rho0 = 1025.0
    omega = 7.2921e-5

    # Coriolis parameter
    f = 2 * omega * np.sin(np.deg2rad(lat))

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    z = ds.depth.values  # m, positive downward

    x = ds.x.values * 1000.0  # km -> m

    lon = ds.lon.values

    temperature = ds.temperature_smooth.values

    salinity = ds.salinity_smooth.values

    nz, nx = temperature.shape

    # --------------------------------------------------------
    # Pressure for TEOS-10
    # --------------------------------------------------------

    p = gsw.p_from_z(-z, lat)

    p_3D = np.broadcast_to(p[:, None], (nz, nx))

    lon_2D = np.broadcast_to(lon[None, :], (nz, nx))

    # --------------------------------------------------------
    # TEOS-10
    # --------------------------------------------------------

    SA = gsw.SA_from_SP(salinity, p_3D, lon_2D, lat)

    CT = gsw.CT_from_t(SA, temperature, p_3D)

    rho = gsw.rho(SA, CT, p_3D)

    # --------------------------------------------------------
    # Horizontal density gradient
    # --------------------------------------------------------

    drhodx = np.full_like(rho, np.nan)

    for k in range(nz):

        valid = np.isfinite(rho[k, :])

        if np.sum(valid) >= 2:

            drhodx[k, valid] = np.gradient(rho[k, valid], x[valid])

    # --------------------------------------------------------
    # Thermal wind shear
    # --------------------------------------------------------

    dv_dz = g / (rho0 * f) * drhodx

    # --------------------------------------------------------
    # Integrate from bottom upward
    # Reference velocity = 0 at bottom
    # --------------------------------------------------------

    vg = np.full_like(rho, np.nan)

    bathymetry = ds.bathymetry.values

    for ix in range(nx):

        zb = bathymetry[ix]

        if not np.isfinite(zb):
            continue

        valid = np.isfinite(dv_dz[:, ix]) & (z <= zb)

        indices = np.where(valid)[0]

        if len(indices) < 2:
            continue

        # Deepest valid level
        k_bottom = indices[-1]

        # Reference velocity
        vg[k_bottom, ix] = 0.0

        # Integrate upward
        for k in range(k_bottom - 1, -1, -1):

            if not np.isfinite(dv_dz[k, ix]):
                continue

            dz = z[k + 1] - z[k]

            vg[k, ix] = vg[k + 1, ix] - 0.5 * (dv_dz[k, ix] + dv_dz[k + 1, ix]) * dz

    # --------------------------------------------------------
    # Output dataset
    # --------------------------------------------------------

    ds_out = ds.copy()

    ds_out["SA"] = (("depth", "x"), SA)

    ds_out["CT"] = (("depth", "x"), CT)

    ds_out["density_TEOS10"] = (("depth", "x"), rho)

    ds_out["drhodx"] = (("depth", "x"), drhodx)

    ds_out["dv_dz"] = (("depth", "x"), dv_dz)

    ds_out["vg"] = (("depth", "x"), vg)

    # --------------------------------------------------------
    # Attributes
    # --------------------------------------------------------

    ds_out["SA"].attrs = {"long_name": "Absolute Salinity", "units": "g kg-1"}

    ds_out["CT"].attrs = {"long_name": "Conservative Temperature", "units": "degC"}

    ds_out["density_TEOS10"].attrs = {
        "long_name": "Seawater density calculated with TEOS-10",
        "units": "kg m-3",
    }

    ds_out["drhodx"].attrs = {"long_name": "Horizontal density gradient", "units": "kg m-4"}

    ds_out["dv_dz"].attrs = {"long_name": "Geostrophic vertical shear", "units": "s-1"}

    ds_out["vg"].attrs = {
        "long_name": "Bottom-referenced meridional geostrophic velocity",
        "units": "m s-1",
        "reference": "Zero velocity at the deepest valid level",
    }

    return ds_out


# ============================================================
# PROCESS ONE NETCDF FILE
# ============================================================


def process_file(input_file, bathy_39, file_number, total_files):

    print("\n")
    print("=" * 70)
    print(f"Processing file {file_number}/{total_files}: " f"{input_file}")
    print("=" * 70)

    ds = xr.open_dataset(input_file)

    try:

        # ====================================================
        # SECTION
        # ====================================================

        trajectory_section, section_length_km = extract_section(ds)

        if len(trajectory_section) == 0:

            print("*****NO PROFILES RETAINED*****" "*****SKIPPING FILE*****")

            return

        # ====================================================
        # BUILD LEGS
        # ====================================================

        ds_legs = build_legs(ds, bathy_39, trajectory_section, section_length_km)

        # ====================================================
        # SMOOTH LEGS
        # ====================================================

        ds_legs_smooth = smooth_legs(ds_legs)

        # ====================================================
        # GEOSTROPHIC VELOCITY
        # ====================================================

        ds_legs_geostrophic = {}

        for leg_number, ds_leg in ds_legs_smooth.items():

            ds_geo = compute_geostrophic_velocity(ds_leg, lat=LAT_SECTION)

            ds_legs_geostrophic[leg_number] = ds_geo

            # ------------------------------------------------
            # Diagnostics
            # ------------------------------------------------

            # vg_min = np.nanmin(
            #     ds_geo.vg.values
            # )

            # vg_max = np.nanmax(
            #     ds_geo.vg.values
            # )

            # print(
            #     f"  Direction: "
            #     f"{ds_leg.attrs['direction']}"
            # )

            # print(
            #     f"  vg min: "
            #     f"{vg_min:.4f} m/s"
            # )

            # print(
            #     f"  vg max: "
            #     f"{vg_max:.4f} m/s"
            # )

        # ====================================================
        # SAVE
        # ====================================================

        output_dir = Path(OUTPUT_DIR)

        output_dir.mkdir(parents=True, exist_ok=True)

        input_stem = Path(input_file).stem

        for leg_number, ds_geo in ds_legs_geostrophic.items():

            direction = ds_geo.attrs.get("direction", "unknown").lower()

            output_file = (
                output_dir / f"{input_stem}" f"_leg_{leg_number:03d}" f"_{direction}" f"_vgeo.nc"
            )

            ds_geo.to_netcdf(output_file)

            print(f"Saved: {output_file}")

    finally:

        ds.close()


# ============================================================
# FIND ALL NETCDF FILES
# ============================================================


def find_netcdf_files():

    input_dir = Path(INPUT_DIR)

    files = sorted(input_dir.rglob("*.nc"))

    return files


# ============================================================
# MAIN
# ============================================================


def main():

    # --------------------------------------------------------
    # Load bathymetry
    # --------------------------------------------------------

    print("Loading bathymetry...")

    bathy_39 = load_bathymetry()

    # --------------------------------------------------------
    # Find NetCDF files
    # --------------------------------------------------------

    netcdf_files = find_netcdf_files()

    print(f"Number of NetCDF files found: " f"{len(netcdf_files)}")

    if len(netcdf_files) == 0:

        print(f"No NetCDF files found in " f"{INPUT_DIR}")

        return

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Process all files
    # --------------------------------------------------------

    for file_number, input_file in enumerate(netcdf_files, start=1):

        try:

            process_file(input_file, bathy_39, file_number, len(netcdf_files))

        except Exception as exc:

            print("\n*****ERROR WHILE PROCESSING*****")

            print(f"  {input_file}")

            print(f"  {type(exc).__name__}: " f"{exc}")

            print("*****CONTINUING WITH THE NEXT FILE*****")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
