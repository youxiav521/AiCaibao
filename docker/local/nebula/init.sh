#!/bin/sh
set -eu

run_ngql() {
    nebula-console \
        -addr "$NEBULA_HOST" \
        -port "$NEBULA_PORT" \
        -u "$NEBULA_USER" \
        -p "$NEBULA_PASSWORD" \
        "$@"
}

storage_online=false
for _ in $(seq 1 60); do
    run_ngql -e "ADD HOSTS \"$NEBULA_STORAGE_HOST\":$NEBULA_STORAGE_PORT;" >/dev/null 2>&1 || true
    if run_ngql -e "SHOW HOSTS STORAGE;" 2>/dev/null | grep -q '"ONLINE"'; then
        storage_online=true
        break
    fi
    sleep 2
done

if [ "$storage_online" != "true" ]; then
    echo "NebulaGraph storage host did not become online." >&2
    exit 1
fi

run_ngql -e "CREATE SPACE IF NOT EXISTS $NEBULA_SPACE (partition_num = 1, replica_factor = 1, vid_type = FIXED_STRING(128));"

space_ready=false
for _ in $(seq 1 60); do
    space_output="$(run_ngql -e "SHOW SPACES;" 2>&1 || true)"
    if printf '%s\n' "$space_output" | grep -q "$NEBULA_SPACE"; then
        space_ready=true
        break
    fi
    sleep 2
done

if [ "$space_ready" != "true" ]; then
    echo "NebulaGraph space did not become ready." >&2
    exit 1
fi

run_ngql -e "CREATE USER IF NOT EXISTS $NEBULA_QUERY_USER WITH PASSWORD '$NEBULA_QUERY_PASSWORD'; GRANT ROLE GUEST ON $NEBULA_SPACE TO $NEBULA_QUERY_USER;"

legacy_tag_output="$(run_ngql -e "USE $NEBULA_SPACE; SHOW TAGS;" 2>&1 || true)"
legacy_edge_output="$(run_ngql -e "USE $NEBULA_SPACE; SHOW EDGES;" 2>&1 || true)"
if printf '%s\n' "$legacy_tag_output" | grep -q '"entity"' ||
    printf '%s\n' "$legacy_edge_output" | grep -q '"relation"'; then
    echo "Legacy entity/relation schema detected. Verify data recovery and recreate $NEBULA_SPACE before initialization." >&2
    exit 1
fi

schema_created=false
for _ in $(seq 1 60); do
    schema_output="$(run_ngql -f /init/schema.ngql 2>&1 || true)"
    if ! printf '%s\n' "$schema_output" | grep -q '\[ERROR'; then
        schema_created=true
        break
    fi
    sleep 2
done

if [ "$schema_created" != "true" ]; then
    printf '%s\n' "$schema_output"
    echo "NebulaGraph schema creation failed." >&2
    exit 1
fi

schema_ready=false
for _ in $(seq 1 60); do
    tag_output="$(run_ngql -e "USE $NEBULA_SPACE; SHOW TAGS;" 2>&1 || true)"
    edge_output="$(run_ngql -e "USE $NEBULA_SPACE; SHOW EDGES;" 2>&1 || true)"
    index_output="$(run_ngql -e "USE $NEBULA_SPACE; SHOW TAG INDEXES;" 2>&1 || true)"

    all_present=true
    for tag in kg_snapshot enterprise person mortgage_asset pledge_asset creditor risk_event project unknown; do
        if ! printf '%s\n' "$tag_output" | grep -q "\"$tag\""; then
            all_present=false
        fi
        if [ "$tag" != "kg_snapshot" ] &&
            ! printf '%s\n' "$index_output" | grep -q "\"i_${tag}_folder_key\""; then
            all_present=false
        fi
    done
    for edge in is_borrower is_guarantor is_pledgor is_mortgagor is_warrantor collateral_for collateral_owned_by spouse_of kinship_of legal_representative executive_of controller_of shareholder_of supplier_of invests_in branch_of loan_to risk_involves credit_from; do
        if ! printf '%s\n' "$edge_output" | grep -q "\"$edge\""; then
            all_present=false
        fi
    done

    if [ "$all_present" = "true" ]; then
        schema_ready=true
        break
    fi
    sleep 2
done

if [ "$schema_ready" != "true" ]; then
    echo "NebulaGraph schema did not become ready." >&2
    exit 1
fi

echo "NebulaGraph initialized: space=$NEBULA_SPACE"
