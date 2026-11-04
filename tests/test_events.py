from pathlib import Path

import pytest

from racfall.evaluation.events import Alert, FallEvent, Video, evaluate, read_alerts, read_events, read_manifest

FIXTURES = Path(__file__).parent / "fixtures"


def video(video_id="v", duration=100.0, has_fall=True):
    return Video(video_id, "sample", video_id, "val", "fall" if has_fall else "ADL", duration, has_fall)


def test_fixture_reference_event_metrics():
    videos = read_manifest(FIXTURES / "manifest.csv")
    events = read_events(FIXTURES / "gt_events.csv", videos)
    predictions = read_alerts(FIXTURES / "alerts.csv", videos)
    r = evaluate(videos, events, predictions)
    assert (r["tp"], r["fn"], r["fp_all_unmatched"]) == (1, 0, 2)
    assert r["event_recall"] == 1
    assert r["event_precision"] == pytest.approx(1 / 3)
    assert r["median_detection_delay_s"] == 1
    assert r["false_alarms_per_adl_hour"] == 2
    assert r["retained_alerts"] == 3


def test_early_alarm_is_not_matched():
    r = evaluate({"v": video()}, [FallEvent("v", "f", 10, 11)], [Alert("v", 9)])
    assert r["tp"] == 0 and r["fn"] == 1 and r["fp_all_unmatched"] == 1
    assert r["median_detection_delay_s"] is None
    assert r["false_alarms_per_adl_hour"] is None


def test_one_alert_cannot_match_two_gt_events():
    v = video()
    r = evaluate({"v": v}, [FallEvent("v", "1", 5, 6), FallEvent("v", "2", 6, 7)], [Alert("v", 7)])
    assert r["tp"] == 1 and r["fn"] == 1


def test_explicit_adl_exposure_denominator_and_refractory():
    v = video("adl", 1800, False)
    r = evaluate({"adl": v}, [], [Alert("adl", 1), Alert("adl", 2), Alert("adl", 12)])
    assert r["adl_false_alerts"] == 2
    assert r["false_alarms_per_adl_hour"] == 4
    assert r["event_recall"] is None


def test_boundary_and_invalid_window():
    v = video()
    r = evaluate({"v": v}, [FallEvent("v", "f", 10, 12)], [Alert("v", 15)])
    assert r["tp"] == 1
    with pytest.raises(ValueError):
        evaluate({"v": v}, [], [], match_window_s=-0.5)


def test_manifest_detects_subject_leakage(tmp_path):
    p = tmp_path / "manifest.csv"
    p.write_text("video_id,dataset,subject_id,split,activity,duration_s,has_fall\n"
                 "x,A,s1,train,ADL,1,0\n"
                 "y,A,s1,test,ADL,1,0\n")
    with pytest.raises(ValueError, match="Subject leakage"):
        read_manifest(p)


def test_missing_gt_onset_refused(tmp_path):
    p = tmp_path / "events.csv"
    p.write_text("video_id,event_id,onset_s,end_s\n")
    with pytest.raises(ValueError, match="Missing onset"):
        read_events(p, {"v": video()})


def test_invalid_input_time_and_unknown_video(tmp_path):
    p = tmp_path / "alerts.csv"
    p.write_text("video_id,alert_s\nv,NaN\n")
    with pytest.raises(ValueError, match="finite"):
        read_alerts(p, {"v": video()})
    p.write_text("video_id,alert_s\nunknown,1\n")
    with pytest.raises(ValueError, match="Unknown"):
        read_alerts(p, {"v": video()})
