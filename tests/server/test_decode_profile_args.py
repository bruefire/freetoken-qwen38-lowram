from freetoken.server.launch import _decode_profile_requested


def test_decode_profile_flag_pre_scan_supports_both_arg_forms() -> None:
    assert _decode_profile_requested(["--decode-profile-interval", "64"])
    assert _decode_profile_requested(["--decode-profile-interval=64"])
    assert not _decode_profile_requested([])
    assert not _decode_profile_requested(["--decode-profile-interval", "0"])
