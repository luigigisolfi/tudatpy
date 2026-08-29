from pydantic import BaseModel, model_validator
from .common import CCSDSHeader


class OMMMetadata(BaseModel):
    """
    Metadata fields specific to Orbit Mean-Elements Messages (OMM).

    See Table 4-2, CCSDS 502.0-B-3 (Orbit Data Messages, April 2023).

    Attributes:
        OBJECT_NAME: Spacecraft name for which mean element data is given.
        OBJECT_ID: Object identifier (recommended: UN OOSA designator).
        CENTER_NAME: Origin of the OMM reference frame (e.g., EARTH).
        REF_FRAME: Reference frame in which the Keplerian elements are
            given. Per 4.2.4.9/4.2.4.6, OMMs derived from NORAD TLEs must
            use "TEME" here (TLE/SGP4's implicit True Equator Mean Equinox
            of Date frame) -- TEME is only valid for TLE-based OMMs.
        REF_FRAME_EPOCH: Epoch of the reference frame, if not intrinsic to
            its definition. Conditional -- only meaningful for some
            REF_FRAME values.
        TIME_SYSTEM: Time system used for the elements (e.g., UTC).
        MEAN_ELEMENT_THEORY: Propagator the mean elements are consistent
            with (e.g., SGP4, SGP4-XP, DSST, USM). Determines which
            TLE-related fields in OMMData are required (see OMMData).
        COMMENT: Optional comments.
    """

    OBJECT_NAME: str
    OBJECT_ID: str
    CENTER_NAME: str = "EARTH"
    REF_FRAME: str
    REF_FRAME_EPOCH: str | None = None
    TIME_SYSTEM: str = "UTC"
    MEAN_ELEMENT_THEORY: str
    COMMENT: str | None = None


class OMMData(BaseModel):
    """
    OMM Data section: Mean Keplerian elements, spacecraft parameters, and
    TLE-related parameters (Table 4-3, CCSDS 502.0-B-3).

    All values are 'at epoch' (4.2.4.3): the value of each parameter at
    the time given by EPOCH, not at message creation time.
    """

    # -- Mean Keplerian Elements (mandatory block) --
    EPOCH: str
    SEMI_MAJOR_AXIS: float | None = None
    MEAN_MOTION: float | None = None
    ECCENTRICITY: float
    INCLINATION: float
    RA_OF_ASC_NODE: float
    ARG_OF_PERICENTER: float
    MEAN_ANOMALY: float
    GM: float | None = None

    # -- Spacecraft Parameters (optional) --
    MASS: float | None = None
    SOLAR_RAD_AREA: float | None = None
    SOLAR_RAD_COEFF: float | None = None
    DRAG_AREA: float | None = None
    DRAG_COEFF: float | None = None

    # -- TLE Related Parameters (only required if MEAN_ELEMENT_THEORY = SGP/SGP4) --
    EPHEMERIS_TYPE: int | None = None
    CLASSIFICATION_TYPE: str | None = None
    NORAD_CAT_ID: int | None = None
    ELEMENT_SET_NO: int | None = None
    REV_AT_EPOCH: int | None = None
    BSTAR: float | None = None
    BTERM: float | None = None
    MEAN_MOTION_DOT: float | None = None
    MEAN_MOTION_DDOT: float | None = None
    AGOM: float | None = None

    @model_validator(mode="after")
    def _check_orbit_size_field(self):
        """
        Table 4-3: SEMI_MAJOR_AXIS is mandatory, "or, if
        MEAN_ELEMENT_THEORY = SGP/SGP4, the Keplerian Mean motion in
        revolutions per day" -- i.e. exactly one orbit-size keyword is
        expected, not necessarily SEMI_MAJOR_AXIS specifically. This only
        checks that at least one was supplied; it does not cross-check
        against MEAN_ELEMENT_THEORY (that would require the owning
        OMMMetadata, which this model does not have a reference to).
        """
        if self.SEMI_MAJOR_AXIS is None and self.MEAN_MOTION is None:
            raise ValueError(
                "OMMData requires SEMI_MAJOR_AXIS or MEAN_MOTION " "(Table 4-3, CCSDS 502.0-B-3)."
            )
        return self


OMM_HEADER_KEYWORDS = (frozenset(CCSDSHeader.model_fields.keys()) - {"CCSDS_VERS", "COMMENT"}) | {
    "CCSDS_OMM_VERS"
}
OMM_METADATA_KEYWORDS = frozenset(OMMMetadata.model_fields.keys()) - {"COMMENT"}
