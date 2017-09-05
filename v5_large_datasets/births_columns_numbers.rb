require 'csv'
require 'json'
require 'finishing_moves'

births_columns_numbers = {}

CSV.foreach('births_columns_counts.csv') do |row|
  column_name, values_hash = row
  values_hash = JSON.parse(values_hash)
  unique_values = values_hash.keys

  values = unique_values.select { |v| v != '' }

  if values.size > 0 && values[0].numeric?
    values = values.map(&:to_f)
    values_count = values_hash.values.reduce(&:+)
    values_sum = values_hash.map do |key, value|
      key.to_f * value
    end.reduce(&:+)

    births_columns_numbers[column_name] = {
      mean: values_sum / values_count,
      min: values.min,
      max: values.max,
    }
  end
end

csv_out = CSV.open('births_columns_numbers.csv', 'wb')

births_columns_numbers.each do |column_name, column_stats|
  csv_out << [column_name, column_stats.to_json]
end