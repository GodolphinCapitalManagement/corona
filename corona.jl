using TOML
using ODBC
using DataFrames

config = TOML.parsefile("/home/gsinha/.config/gcm/gcm.toml")

function make_conn(config, database)
    user = config["db"]["db_user"]
    pswd = config["db"]["db_pswd"]
    db_conn_string = string(
        "Driver={PostgreSQL Unicode};Server=127.0.0.1;Port=5432;Database=$database;",
        "Uid=$user;Pwd=$pswd;"
    )

    ODBC.Connection(db_conn_string)
end

conn = make_conn(config, "pympl")
@time df = DBInterface.execute(conn, "select * from consumer.csummary;") |> DataFrame

println(describe(df))
