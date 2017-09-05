require 'csv'
require 'json'
require 'finishing_moves'

# store all the nans columns just to complete all the columns
births_columns_nans = {}

CSV.foreach('births_columns_counts.csv') do |row|
  column_name, values_hash = row
  values_hash = JSON.parse(values_hash)
  unique_values = values_hash.keys

  values = unique_values.select { |v| v != '' }

  if values.size == 0
    births_columns_nans[column_name] ||= {}
  end
end

csv_out = CSV.open('births_columns_nans.csv', 'wb')

births_columns_nans.each do |column_name, column_nan|
  csv_out << [column_name, column_nan.to_json]
end