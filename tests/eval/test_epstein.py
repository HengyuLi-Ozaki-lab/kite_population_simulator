import json

import pandas as pd
import pytest

from kite.eval import epstein


def write_csv(path, rows):
    columns = sorted({k for r in rows for k in r})
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def rows_for(pid, wave, treatment, completed=1, **covariates):
    base = {
        "id": pid,
        "wave": wave,
        "treatment": treatment,
        "completed": completed,
        "age": 34,
        "gender": 2,
        "white": 1,
        "college": 1,
        "education": 6,
        "demrep_c": 2,
        "region": 3,
        "hispanic": 1,
    }
    base.update(covariates)
    return [{**base, "item_num": k, "real": int(k > 10), "rating": k % 2} for k in range(1, 21)]


@pytest.fixture
def csv(tmp_path):
    rows = (
        rows_for(1, 3, 1.0)
        + rows_for(2, 3, 5.0, age=71, gender=1, white=0, education=2, demrep_c=6, region=1, hispanic=2)
        + rows_for(3, 3, 2.0, completed=0)
    )
    rows += rows_for(4, 4, 4.5, age=12, gender=5, demrep_c=float("nan"), education=11)
    path = tmp_path / "data_to_post.csv"
    write_csv(path, rows)
    return path


def test_design_keeps_completed_participants_and_names_the_arms(csv):
    design = epstein.load_design(csv)
    assert len(design) == 3 and 3 not in set(design["id"])
    assert dict(zip(design["id"], design["arm"], strict=True)) == {1: "control", 2: "tips", 4: "partisan_norms"}
    assert "rating" not in design.columns and "item_num" not in design.columns


def test_ratings_are_refused_until_the_criteria_are_committed(csv, tmp_path):
    with pytest.raises(PermissionError):
        epstein.load_ratings(csv, criteria=tmp_path / "not_committed.yaml")


def test_ratings_shape_when_allowed(csv, monkeypatch):
    monkeypatch.setattr(epstein, "_committed", lambda path: True)
    ratings = epstein.load_ratings(csv, criteria=csv)
    assert len(ratings) == 60 and set(ratings["arm"]) == {"control", "tips", "partisan_norms"}
    assert ratings.groupby("id")["true"].sum().tolist() == [10, 10, 10]


def test_persona_uses_words_and_drops_invalid_values(csv):
    design = epstein.load_design(csv).set_index("id")
    assert epstein.render_persona(design.loc[1]) == {
        "age": "in their 30s", "gender": "female", "race": "white", "hispanic_or_latino": "no",
        "education": "a bachelor's degree", "party": "a Democrat", "region_of_the_us": "the South",
    }  # fmt: skip
    older = epstein.render_persona(design.loc[2])
    assert older["age"] == "in their 70s" and older["gender"] == "male" and older["race"] == "not white"
    assert older["hispanic_or_latino"] == "yes" and older["party"] == "a strong Republican" and older["education"] == "a high school diploma"
    odd = epstein.render_persona(design.loc[4])
    assert "age" not in odd and odd["gender"] == "another gender" and "party" not in odd and "education" not in odd


def test_headlines_table(tmp_path):
    path = tmp_path / "headlines.json"
    path.write_text(
        json.dumps(
            {
                "provenance": "test",
                "items": {"1": {"source": "A.COM", "text": "x", "true": False}, "11": {"source": "B.COM", "text": "y", "true": True}},
            }
        )
    )
    table = epstein.headlines(path)
    assert table[1] == {"source": "A.COM", "text": "x", "true": False} and table[11]["true"] is True


def test_screens_resolve_refs_and_build_contexts():
    screens = epstein.load_screens()
    assert screens["screens"]["tips_norms"][0] == screens["screens"]["partisan_norms"][0]
    assert screens["screens"]["tips_norms"][1] == screens["screens"]["tips"][0]
    control = epstein.context_for("control", screens)
    assert control.startswith('The survey began: "You will be presented') and "sharing the information" in control
    evaluation = epstein.context_for("evaluation", screens)
    assert "pretest" in evaluation and "neutron star" in evaluation and "true news headline" in evaluation
    assert evaluation.endswith('Next the survey said: "' + screens["instruction"]["sharing"] + '"')
    tips_norms = epstein.context_for("tips_norms", screens)
    assert tips_norms.index("8 out of 10") < tips_norms.index("Be skeptical") and tips_norms.count("The survey began") == 1
    assert "The survey continued" in tips_norms
    assert epstein.context_for("accuracy_only", screens).endswith('information accurate."')
    text, options = epstein.question_for("control", screens)
    assert text.startswith("Would you consider sharing") and options == ["No", "Yes"]
    with pytest.raises(KeyError):
        epstein.context_for("not_an_arm", screens)


def test_task_items_pair_every_arm_of_a_wave_per_persona(csv, tmp_path):
    path = tmp_path / "headlines.json"
    items = {str(k): {"source": "X.COM", "text": f"headline {k}", "true": k > 10} for k in (1, 2, 11, 12)}
    path.write_text(json.dumps({"provenance": "test", "items": items}))
    task = epstein.EpsteinTask(csv, personas_per_wave=1, audit_per_wave=1, headlines_path=path)
    assert task.arms == {3: ["control", "tips"], 4: ["partisan_norms"]}
    listed = list(task.items())
    ids = [i.item_id for i in listed]
    assert len(ids) == len(set(ids))
    wave3 = [i for i in listed if i.meta["wave"] == 3]
    assert {i.meta["pool"] for i in wave3} == {"application", "audit"} and len(wave3) == 2 * 2 * 4  # 2 personas x 2 arms x 4 cards
    one = next(i for i in wave3 if i.meta["arm"] == "tips")
    assert one.request.state["survey"]["context"].startswith("The survey began with a screen reading")
    assert "headline" in one.request.state["survey"]["question"] and one.request.state["respondent"]["age"] in ("in their 30s", "in their 70s")
