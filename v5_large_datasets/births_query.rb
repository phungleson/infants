require 'sqlite3'
require 'csv'

unique_values = {}

begin
  db = SQLite3::Database.open 'infants.sqlite'

  columns = db.execute "PRAGMA table_info(infants_births)"
  column_names = columns.map { |column| column[1] }
  column_names.each do |column_name|
    next if column_name == 'index'
    # next if column_name == 'idnumber'
    puts column_name

    distinct_values = db.execute "SELECT DISTINCT `#{column_name}` FROM infants_births"
    distinct_values = distinct_values.map { |v| v[0] }
    unique_values[column_name] = distinct_values
  end
ensure
  db.close if db
end

csv = CSV.open('infants_columns_sqlite.csv', 'w')

unique_values.sort_by do |k, values|
  values.size
end.each do |k, values|
  csv << [k, values]
end